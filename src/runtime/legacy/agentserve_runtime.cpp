// Single-engine dual submission-path shared-KV serving runtime (Qwen2.5-3B).
// ONE llama_context (shared KV). Prefill path -> stream0 (prefill green ctx); decode path -> stream1
// (decode green ctx). Reads sessions.txt (sid|b64(cold)|d0|b64(app)|d1|...), greedy-generates, records.
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
// minimal base64 decode
static std::string b64d(const std::string& in){
    static const std::string T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; int val=0, bits=0;
    for(char c: in){ if(c=='=') break; auto p=T.find(c); if(p==std::string::npos) continue; val=(val<<6)|p; bits+=6; if(bits>=8){ bits-=8; out+=(char)((val>>bits)&0xff); } }
    return out;
}
static int decode_batch(llama_context*ctx,llama_batch&b,int stream){ std::lock_guard<std::mutex> lk(g_lock); ggml_cuda_as_set_stream(stream); return llama_decode(ctx,b); }

int main(int argc,char**argv){
    std::string trace = argc>1?argv[1]:"/root/autodl-tmp/exp/traces/sessions.txt";
    int nsess = argc>2?atoi(argv[2]):3;
    g_ev.open("/root/autodl-tmp/exp/raw_logs/agentserve_runtime_events.jsonl");
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=99;
    llama_model* model=llama_model_load_from_file("/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf",mp);
    if(!model){ fprintf(stderr,"load FAIL\n"); return 1; }
    llama_context_params cp=llama_context_default_params(); cp.n_ctx=8192; cp.n_batch=4096; cp.n_threads=4; cp.n_seq_max=1;
    llama_context* ctx=llama_init_from_model(model,cp);
    if(!ctx){ fprintf(stderr,"ctx FAIL\n"); return 1; }
    const llama_vocab* v = llama_model_get_vocab(model); int nv=llama_vocab_n_tokens(v);
    std::vector<llama_token> toks(16384);
    std::string line; int nl=0; int sess_done=0; double total_tok=0;
    while(std::getline(std::ifstream(trace),line) && sess_done<nsess){
        if(line.empty()) continue; nl++;
        std::vector<std::string> parts; std::stringstream ss(line); std::string p;
        while(std::getline(ss,p,'|')) parts.push_back(p);
        if(parts.size()<3) continue;
        std::string sid=parts[0];
        // first prefill: b64(cold), then decode count
        std::string cold=b64d(parts[1]); int d0=atoi(parts[2].c_str());
        double treq=now_ms();
        int nt=llama_tokenize(v,cold.c_str(),(int)cold.size(),toks.data(),(int)toks.size(),true,true);
        llama_batch pb=llama_batch_init(nt,0,1);
        for(int i=0;i<nt;i++){ pb.token[i]=toks[i]; pb.pos[i]=i; pb.n_seq_id[i]=1; pb.seq_id[i][0]=0; pb.logits[i]=(i==nt-1); } pb.n_tokens=nt;
        int rc=decode_batch(ctx,pb,0); llama_batch_free(pb);
        double ttft=now_ms()-treq; ev("TTFT",sid,0,ttft);
        int n_past=nt;
        // generate d0 greedy decode on stream1 (explicit batch: pos=n_past, seq 0)
        std::vector<double> tg; llama_batch db0=llama_batch_init(1,0,1);
        for(int i=0;i<d0;i++){ const float*L=llama_get_logits(ctx); int t=argmax(L,nv);
            db0.token[0]=t; db0.pos[0]=n_past; db0.n_seq_id[0]=1; db0.seq_id[0][0]=0; db0.logits[0]=1; db0.n_tokens=1;
            double a=now_ms(); int rc2=decode_batch(ctx,db0,1); double b=now_ms(); tg.push_back(b-a); n_past++; total_tok++; }
        llama_batch_free(db0);
        double tpot0 = d0>0? (tg.empty()?0:0) : 0; double s=0; for(double x:tg) s+=x; tpot0 = d0>0? s/d0:0; ev("TPOT",sid,0,tpot0);
        fprintf(stderr,"[runtime] %s cold n=%d rc=%d TTFT=%.1fms decode d0=%d TPOT=%.1fms\n",sid.c_str(),nt,rc,ttft,d0,tpot0);
        // resume appends
        for(size_t j=3;j+1<parts.size();j+=2){
            std::string app=b64d(parts[j]); int dj=atoi(parts[j+1].c_str());
            int na=llama_tokenize(v,app.c_str(),(int)app.size(),toks.data(),(int)toks.size(),false,false);
            llama_batch rb=llama_batch_init(na,0,1);
            for(int i=0;i<na;i++){ rb.token[i]=toks[i]; rb.pos[i]=n_past+i; rb.n_seq_id[i]=1; rb.seq_id[i][0]=0; rb.logits[i]=(i==na-1); } rb.n_tokens=na;
            int rj=decode_batch(ctx,rb,0); llama_batch_free(rb); n_past+=na;
            llama_batch dbj=llama_batch_init(1,0,1);
            for(int i=0;i<dj;i++){ const float*L=llama_get_logits(ctx); int t=argmax(L,nv);
                dbj.token[0]=t; dbj.pos[0]=n_past; dbj.n_seq_id[0]=1; dbj.seq_id[0][0]=0; dbj.logits[0]=1; dbj.n_tokens=1;
                decode_batch(ctx,dbj,1); n_past++; total_tok++; }
            llama_batch_free(dbj);
            fprintf(stderr,"[runtime] %s resume n=%d rc=%d decode=%d\n",sid.c_str(),na,rj,dj);
        }
        sess_done++;
    }
    llama_free(ctx); llama_model_free(model); g_ev.close();
    fprintf(stderr,"[runtime] DONE sessions=%d total_decode_tokens=%.0f\n",sess_done,total_tok);
    return 0;
}
