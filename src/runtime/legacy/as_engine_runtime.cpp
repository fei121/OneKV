// Single-engine shared-KV serving runtime: TWO contexts (A=prefill, B=decode) sharing one KV pool.
// B is created with ctx_other=A -> shares k/v tensors + cells. A writes prefill KV, B reads+appends.
// Sequential per-session first. Reads sessions.txt: sid|coldb64|d0|appb64|d1|...
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>
extern "C" void ggml_cuda_as_set_stream(int n);
static std::mutex g_lock;
static std::ofstream g_ev;
static double now_ms(){ using namespace std::chrono; return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count()/1e3; }
static void ev(const std::string&e,const std::string&sid,int ph,double ms){ g_ev<<"{\"event\":\""<<e<<"\",\"session\":\""<<sid<<"\",\"phase\":"<<ph<<",\"ms\":"<<ms<<"}\n"; g_ev.flush(); }
static int argmax(const float*L,int n){ int b=0; for(int i=1;i<n;i++) if(L[i]>L[b]) b=i; return b; }
static std::string b64d(const std::string& in){
    static const std::string T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; int val=0, bits=0;
    for(char c: in){ if(c=='=') break; auto p=T.find(c); if(p==std::string::npos) continue; val=(val<<6)|p; bits+=6; if(bits>=8){ bits-=8; out+=(char)((val>>bits)&0xff); } }
    return out;
}
static int dec_batch(llama_context*ctx,llama_batch&b,int stream){ std::lock_guard<std::mutex> lk(g_lock); ggml_cuda_as_set_stream(stream); int r=llama_decode(ctx,b); cudaDeviceSynchronize(); return r; }

int main(int argc,char**argv){
    std::string trace = argc>1?argv[1]:"/root/autodl-tmp/exp/traces/sessions.txt";
    int nsess = argc>2?atoi(argv[2]):3;
    g_ev.open("/root/autodl-tmp/exp/raw_logs/as_engine_runtime_events.jsonl");
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=99;
    llama_model* model=llama_model_load_from_file("/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf",mp);
    if(!model){ fprintf(stderr,"load FAIL\n"); return 1; }
    // context A = prefill engine (primary, owns KV)
    llama_context_params cpa=llama_context_default_params(); cpa.n_ctx=8192; cpa.n_batch=4096; cpa.n_threads=4; cpa.n_seq_max=1;
    llama_context* A=llama_init_from_model(model,cpa);
    if(!A){ fprintf(stderr,"A FAIL\n"); return 1; }
    // context B = decode engine, created BEFORE any prefill (order matters), shares A KV
    llama_context_params cpb=llama_context_default_params(); cpb.n_ctx=8192; cpb.n_batch=4096; cpb.n_threads=4; cpb.n_seq_max=1; cpb.ctx_other=A;
    llama_context* B=llama_init_from_model(model,cpb);
    if(!B){ fprintf(stderr,"B FAIL\n"); return 1; }
    const llama_vocab* v = llama_model_get_vocab(model); int nv=llama_vocab_n_tokens(v);
    std::vector<llama_token> toks(16384);
    std::string line; int sess_done=0; double total_tok=0;
    std::ifstream ifs(trace);
    while(std::getline(ifs,line) && sess_done<nsess){
        if(line.empty()) continue;
        std::vector<std::string> parts; std::stringstream ss(line); std::string p;
        while(std::getline(ss,p,'|')) parts.push_back(p);
        if(parts.size()<3) continue;
        std::string sid=parts[0];
        std::string cold=b64d(parts[1]); int d0=atoi(parts[2].c_str());
        double treq=now_ms();
        int nt=llama_tokenize(v,cold.c_str(),(int)cold.size(),toks.data(),(int)toks.size(),true,true);
        llama_batch pb=llama_batch_init(nt,0,1);
        for(int i=0;i<nt;i++){ pb.token[i]=toks[i]; pb.pos[i]=i; pb.n_seq_id[i]=1; pb.seq_id[i][0]=0; pb.logits[i]=(i==nt-1); } pb.n_tokens=nt;
        int rc=dec_batch(A,pb,0); llama_batch_free(pb);
        double ttft=now_ms()-treq; ev("TTFT",sid,0,ttft);
        int n_past=nt;
        // decode d0 on B (decode engine), seeded from A prefill logits
        const float* LA=llama_get_logits(A); int tok=argmax(LA,nv);
        std::vector<double> tg; llama_batch db=llama_batch_init(1,0,1);
        for(int i=0;i<d0;i++){
            db.token[0]=tok; db.pos[0]=n_past; db.n_seq_id[0]=1; db.seq_id[0][0]=0; db.logits[0]=1; db.n_tokens=1;
            double a=now_ms(); int rc2=dec_batch(B,db,1); double b=now_ms(); tg.push_back(b-a); n_past++; total_tok++;
            const float* L=llama_get_logits(B); tok=argmax(L,nv);
        }
        llama_batch_free(db);
        double tpot0 = d0>0? [&]{double s=0;for(double x:tg)s+=x;return s/d0;}() : 0; ev("TPOT",sid,0,tpot0);
        fprintf(stderr,"[as] %s cold n=%d rc=%d TTFT=%.1fms decode d0=%d TPOT=%.1fms next=%d\n",sid.c_str(),nt,rc,ttft,d0,tpot0,tok);
        // resume appends: prefill on A, decode on B
        for(size_t j=3;j+1<parts.size();j+=2){
            std::string app=b64d(parts[j]); int dj=atoi(parts[j+1].c_str());
            int na=llama_tokenize(v,app.c_str(),(int)app.size(),toks.data(),(int)toks.size(),false,false);
            llama_batch rb=llama_batch_init(na,0,1);
            for(int i=0;i<na;i++){ rb.token[i]=toks[i]; rb.pos[i]=n_past+i; rb.n_seq_id[i]=1; rb.seq_id[i][0]=0; rb.logits[i]=(i==na-1); } rb.n_tokens=na;
            double ra=now_ms(); int rj=dec_batch(A,rb,0); double rbms=now_ms()-ra; llama_batch_free(rb); n_past+=na;
            ev("ResumeTTFT",sid,0,rbms);
            const float* LA2=llama_get_logits(A); tok=argmax(LA2,nv);
            std::vector<double> tgr; llama_batch dbj=llama_batch_init(1,0,1);
            for(int i=0;i<dj;i++){ dbj.token[0]=tok; dbj.pos[0]=n_past; dbj.n_seq_id[0]=1; dbj.seq_id[0][0]=0; dbj.logits[0]=1; dbj.n_tokens=1;
                double a=now_ms(); int rc3=dec_batch(B,dbj,1); double b=now_ms(); tgr.push_back(b-a); n_past++; total_tok++;
                const float* L=llama_get_logits(B); tok=argmax(L,nv); }
            llama_batch_free(dbj);

            // simpler:
            double tpot1=0; { double s2=0; for(double x:tgr) s2+=x; tpot1 = dj>0? s2/dj:0; }
            ev("TPOT_resume",sid,0,tpot1);
            fprintf(stderr,"[as] %s resume n=%d rc=%d resumeTTFT=%.1fms decode=%d TPOT=%.1fms\n",sid.c_str(),na,rj,rbms,dj,tpot1);
        }
        sess_done++;
    }
    llama_free(B); llama_free(A); llama_model_free(model); g_ev.close();
    fprintf(stderr,"[as] DONE sessions=%d total_decode_tokens=%.0f\n",sess_done,total_tok);
    return 0;
}
