// Single-engine shared-KV serving: N sessions, A=prefill engine (green0), B=decode engine (green1, shares A KV).
// Reads sessions.txt (sid|coldb64|d0|appb64|d1|...), serves N agents, records TTFT + TPOT.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>
extern "C" void ggml_cuda_as_set_stream(int n);
static std::mutex g_lock;
static std::ofstream g_ev;
static double now_ms(){ using namespace std::chrono; return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count()/1e3; }
static void ev(const std::string&e,const std::string&sid,double ms){ g_ev<<"{\"event\":\""<<e<<"\",\"session\":\""<<sid<<"\",\"ms\":"<<ms<<"}\n"; g_ev.flush(); }
static int argmax(const float*L,int n){ int b=0; for(int i=1;i<n;i++) if(L[i]>L[b]) b=i; return b; }
static std::string b64d(const std::string& in){ static const std::string T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; std::string out; int val=0,bits=0; for(char c:in){ if(c=='=')break; auto p=T.find(c); if(p==std::string::npos)continue; val=(val<<6)|p; bits+=6; if(bits>=8){ bits-=8; out+=(char)((val>>bits)&0xff);} } return out; }
struct Session {
    std::string sid; int seq=0; std::vector<std::string> parts; size_t idx=0; int n_past=0; bool done=false;
};
int main(int argc,char**argv){
    std::string trace=argc>1?argv[1]:"/root/autodl-tmp/exp/traces/sessions.txt";
    int N=argc>2?atoi(argv[2]):3;
    g_ev.open("/root/autodl-tmp/exp/raw_logs/as_serve_events.jsonl");
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=99;
    llama_model* model=llama_model_load_from_file("/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf",mp);
    llama_context_params cpa=llama_context_default_params(); cpa.n_ctx=8192; cpa.n_batch=4096; cpa.n_threads=4; cpa.n_seq_max=N;
    llama_context* A=llama_init_from_model(model,cpa);
    llama_context_params cpb=llama_context_default_params(); cpb.n_ctx=8192; cpb.n_batch=4096; cpb.n_threads=4; cpb.n_seq_max=N; cpb.ctx_other=A;
    llama_context* B=llama_init_from_model(model,cpb);
    const llama_vocab* v=llama_model_get_vocab(model); int nv=llama_vocab_n_tokens(v);
    std::vector<llama_token> toks(16384);
    std::vector<Session> ses(N);
    { std::ifstream ifs(trace); std::string line; int i=0; while(std::getline(ifs,line)&&i<N){ if(line.empty())continue;
        std::stringstream ss(line); std::string p; std::vector<std::string> parts; while(std::getline(ss,p,'|'))parts.push_back(p);
        ses[i].sid=parts[0]; ses[i].parts=parts; ses[i].seq=i; i++; } }
    auto prefill=[&](Session&s, bool cold){
        std::string text = cold? b64d(s.parts[1]) : b64d(s.parts[1+2*(s.idx-1)]);
        int nt=llama_tokenize(v,text.c_str(),(int)text.size(),toks.data(),(int)toks.size(),cold,true);
        llama_batch pb=llama_batch_init(nt,0,1);
        for(int i=0;i<nt;i++){ pb.token[i]=toks[i]; pb.pos[i]=s.n_past+i; pb.n_seq_id[i]=1; pb.seq_id[i][0]=s.seq; pb.logits[i]=(i==nt-1);} pb.n_tokens=nt;
        std::lock_guard<std::mutex> lk(g_lock); ggml_cuda_as_set_stream(0); double t0=now_ms(); int r=llama_decode(A,pb); cudaDeviceSynchronize(); double el=now_ms()-t0; llama_batch_free(pb);
        s.n_past+=nt; (void)r; return el;
    };
    auto decode_all=[&](Session&s, int dcount)->double{
        const float* L=llama_get_logits(A); int tok=argmax(L,nv); double sum=0; int cnt=0;
        llama_batch db=llama_batch_init(1,0,1);
        for(int k=0;k<dcount;k++){
            db.token[0]=tok; db.pos[0]=s.n_past; db.n_seq_id[0]=1; db.seq_id[0][0]=s.seq; db.logits[0]=1; db.n_tokens=1;
            std::lock_guard<std::mutex> lk(g_lock); ggml_cuda_as_set_stream(1); double a=now_ms(); int r=llama_decode(B,db); cudaDeviceSynchronize(); double b=now_ms(); llama_batch_free(db); (void)r;
            s.n_past++; sum+=b-a; cnt++;
            const float* LB=llama_get_logits(B); tok=argmax(LB,nv);
            db=llama_batch_init(1,0,1);
        }
        return cnt>0? sum/cnt:0;
    };
    // cold prefill all sessions (green0)
    for(int i=0;i<N;i++){ double el=prefill(ses[i],true); ev("TTFT_cold",ses[i].sid,el); fprintf(stderr,"[serve] %s cold TTFT=%.1fms\n",ses[i].sid.c_str(),el); }
    // phase loop: decode then resume for each session until done
    int guard=0; bool any=true;
    while(any && guard<100){ any=false; guard++;
        for(int i=0;i<N;i++){ Session&s=ses[i]; if(s.done)continue;
            int dcount = s.idx==0? atoi(s.parts[2].c_str()) : atoi(s.parts[2+2*(s.idx-1)].c_str());
            double tpot=decode_all(s,dcount); ev("TPOT",s.sid,tpot); fprintf(stderr,"[serve] %s phase%d decode=%d TPOT=%.1fms\n",s.sid.c_str(),(int)s.idx,dcount,tpot);
            s.idx++;
            if(s.idx*2 < s.parts.size()){ double el=prefill(s,false); ev("TTFT_resume",s.sid,el); fprintf(stderr,"[serve] %s resume TTFT=%.1fms\n",s.sid.c_str(),el); any=true; }
            else { s.done=true; }
        }
        if(!any) break;
    }
    llama_free(B); llama_free(A); llama_model_free(model); g_ev.close();
    fprintf(stderr,"[serve] DONE sessions=%d\n",N);
    return 0;
}
