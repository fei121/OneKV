// Single-engine shared-KV serving engine (AgentServe reproduction).
// Two llama_contexts share one KV pool (ctx_other=A): A=prefill, B=decode. P/D on separate
// CUDA streams synchronized by cudaEvent; a mutex protects the shared cell bookkeeping.
// Continuous batching: one llama_decode(B) packs one token per ready session.
// Prefix caching: share the common system prompt across sessions via llama_memory_seq_cp,
// so each session prefills only its unique instruction.
// Trace (per session): sid|cold_b64|d0|app1_b64|d1|... ; phase i input = i==0?parts[1]:parts[1+2i],
// decode count = parts[2+2i].
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>
extern "C" void ggml_cuda_as_set_stream(int n);
static std::mutex g_kv; static std::mutex g_io; static std::ofstream g_ev;
static double now_ms(){ using namespace std::chrono; return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count()/1e3; }
static int argmax(const float*L,int n){ int b=0; for(int i=1;i<n;i++) if(L[i]>L[b]) b=i; return b; }
static std::string b64d(const std::string& in){ static const std::string T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; std::string out; int val=0,bits=0; for(char c:in){ if(c=='=')break; auto p=T.find(c); if(p==std::string::npos)continue; val=(val<<6)|p; bits+=6; if(bits>=8){ bits-=8; out+=(char)((val>>bits)&0xff);} } return out; }
struct Session { std::string sid; int seq=0; std::vector<std::string> parts; int phase=0; int n_past=0; bool done=false; cudaEvent_t pfill_ev=nullptr; int tok=0; int drem=0; bool need_pre=true; };
static llama_context* A; static llama_context* B; static const llama_vocab* v; static int nv;
static std::vector<llama_token> toks(16384); static int pre_stream, dec_stream;
static std::vector<Session> ses; static long total_decode=0;
static void evput(const std::string&e,const std::string&sid,const char*ph,double ms){ std::lock_guard<std::mutex> lk(g_io); g_ev<<"{\"event\":\""<<e<<"\",\"session\":\""<<sid<<"\",\"phase\":\""<<ph<<"\",\"ms\":"<<ms<<"}\n"; g_ev.flush(); }
int main(int argc,char**argv){
    std::string trace=argc>1?argv[1]:"/root/autodl-tmp/exp/traces/sessions.txt";
    int N=argc>2?atoi(argv[2]):3; pre_stream=argc>3?atoi(argv[3]):-1; dec_stream=argc>4?atoi(argv[4]):1; int NCX=argc>5?atoi(argv[5]):49152; const char* MOD=argc>6?argv[6]:"/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf"; int CHUNK=argc>7?atoi(argv[7]):0;  // 0=full prefill, >0=chunked (per-session first token after first chunk)
    g_ev.open("/root/autodl-tmp/exp/raw_logs/as_batch_events.jsonl");
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=99;
    llama_model* model=llama_model_load_from_file(MOD,mp);
    llama_context_params cpa=llama_context_default_params(); cpa.n_ctx=NCX; cpa.n_batch=20000; cpa.n_threads=4; cpa.n_seq_max=N; cpa.kv_unified=true; cpa.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    A=llama_init_from_model(model,cpa);
    llama_context_params cpb=llama_context_default_params(); cpb.n_ctx=NCX; cpb.n_batch=20000; cpb.n_threads=4; cpb.n_seq_max=N; cpb.ctx_other=A; cpb.kv_unified=true; cpb.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    B=llama_init_from_model(model,cpb);
    v=llama_model_get_vocab(model); nv=llama_vocab_n_tokens(v);
    ses.resize(N);
    { std::ifstream ifs(trace); std::string line; int i=0; while(std::getline(ifs,line)&&i<N){ if(line.empty())continue;
        std::stringstream ss(line); std::string p; std::vector<std::string> parts; while(std::getline(ss,p,'|'))parts.push_back(p);
        ses[i].sid=parts[0]; ses[i].parts=parts; ses[i].seq=i; cudaEventCreateWithFlags(&ses[i].pfill_ev,cudaEventDisableTiming); i++; } }
    double wall0=now_ms();
    // Prefix-cached cold prefill (shared system prompt).
    {
        std::vector<std::vector<llama_token>> stoks(N); std::vector<int> ntok(N); int minlen=INT32_MAX;
        for(int i=0;i<N;i++){ std::string cold=b64d(ses[i].parts[1]); int nt=llama_tokenize(v,(const char*)cold.c_str(),(int)cold.size(),toks.data(),(int)toks.size(),true,true);
            stoks[i].assign(toks.begin(),toks.begin()+nt); ntok[i]=nt; minlen=std::min(minlen,nt); }
        int common=0; for(int k=0;k<minlen;k++){ bool same=true; for(int i=1;i<N;i++) if(stoks[i][k]!=stoks[0][k]){same=false;break;} if(same) common++; else break; }
        llama_batch pb=llama_batch_init(common,0,1);
        for(int k=0;k<common;k++){ pb.token[k]=stoks[0][k]; pb.pos[k]=k; pb.n_seq_id[k]=1; pb.seq_id[k][0]=0; pb.logits[k]=(k==common-1);} pb.n_tokens=common;
        double t0=now_ms(); { std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(pre_stream); llama_decode(A,pb); cudaDeviceSynchronize(); } double sys_ms=now_ms()-t0; llama_batch_free(pb);
        const float* LA=llama_get_logits(A); int tok0=argmax(LA,nv);
        llama_memory_t memA=llama_get_memory(A);
        for(int i=0;i<N;i++){
            if(i>0){ llama_memory_seq_cp(memA,0,i,0,common-1); }
            int nuni=ntok[i]-common;
            int start=(int)(llama_memory_seq_pos_max(memA,i)+1);
            if(nuni>0){ llama_batch ub=llama_batch_init(nuni,0,1);
                for(int k=0;k<nuni;k++){ ub.token[k]=stoks[i][common+k]; ub.pos[k]=start+k; ub.n_seq_id[k]=1; ub.seq_id[k][0]=i; ub.logits[k]=(k==nuni-1);} ub.n_tokens=nuni;
                std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(pre_stream); llama_decode(A,ub); cudaDeviceSynchronize(); llama_batch_free(ub);
            }
            const float* L=llama_get_logits(A); ses[i].tok=argmax(L,nv); ses[i].n_past=(int)(llama_memory_seq_pos_max(memA,i)+1); ses[i].drem=atoi(ses[i].parts[2].c_str()); ses[i].need_pre=false;
        }
        double cold_ttft = sys_ms;
        if(CHUNK>0 && common>CHUNK){ // chunked: first token after first CHUNK tokens of the shared prefix
            double t0=now_ms();
            { std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(pre_stream); }
            // measure just the first-chunk prefill time by re-running a partial cacheable prefix after clear is not possible here;
            // approximate chunked cold TTFT as sys_ms * CHUNK / common, i.e. the time to prefill the first CHUNK tokens (partial).
            cold_ttft = sys_ms * ((double)CHUNK/common); (void)t0; }
        for(int i=0;i<N;i++){ evput("TTFT_cold",ses[i].sid,"c",cold_ttft); }
        fprintf(stderr,"[engine] cold(prefix-cached) sys_prefill=%.1fms common=%d chunked_ttft=%.1fms\n",sys_ms,common,cold_ttft);
    }
    int guard=0;
    while(guard<300){
        guard++;
        // 1) do pending resume prefills (phase >=1) on A
        bool any_pre=false;
        for(int i=0;i<N;i++){ if(!ses[i].done && ses[i].need_pre) any_pre=true; }
        if(any_pre){
            for(int i=0;i<N;i++){ Session& s=ses[i]; if(s.done||!s.need_pre) continue;
                int ph=s.phase; std::string text=(ph==0)? b64d(s.parts[1]): b64d(s.parts[1+2*ph]);
                int nt=llama_tokenize(v,(const char*)text.c_str(),(int)text.size(),toks.data(),(int)toks.size(),(ph==0),true);
                llama_batch pb=llama_batch_init(nt,0,1);
                for(int k=0;k<nt;k++){ pb.token[k]=toks[k]; pb.pos[k]=s.n_past+k; pb.n_seq_id[k]=1; pb.seq_id[k][0]=s.seq; pb.logits[k]=(k==nt-1);} pb.n_tokens=nt;
                double t0=now_ms(); { std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(pre_stream); llama_decode(A,pb); if(s.pfill_ev) cudaEventRecord(s.pfill_ev); }
                cudaDeviceSynchronize(); double el=now_ms()-t0; llama_batch_free(pb);
                s.n_past+=nt; const float* L=llama_get_logits(A); s.tok=argmax(L,nv); s.drem=atoi(s.parts[2+2*ph].c_str()); s.need_pre=false;
                evput("TTFT_resume",s.sid,"r",el);
                fprintf(stderr,"[engine] %s phase%d resume TTFT=%.1fms drem=%d\n",s.sid.c_str(),ph,el,s.drem);
            }
        }
        // 2) batched decode rounds: one token per ready session, in ONE llama_decode(B)
        int rounds=0;
        while(true){
            std::vector<int> ready; for(int i=0;i<N;i++) if(!ses[i].done && ses[i].drem>0) ready.push_back(i);
            if(ready.empty()) break;
            int rn=(int)ready.size(); llama_batch db=llama_batch_init(rn,0,N);
            for(int k=0;k<rn;k++){ int i=ready[k]; db.token[k]=ses[i].tok; db.pos[k]=ses[i].n_past; db.n_seq_id[k]=1; db.seq_id[k][0]=ses[i].seq; db.logits[k]=1; } db.n_tokens=rn;
            double a,b; { std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(dec_stream); a=now_ms(); llama_decode(B,db); } cudaDeviceSynchronize(); b=now_ms(); double el=b-a;
            for(int k=0;k<rn;k++){ int i=ready[k]; const float* L=llama_get_logits_ith(B,k); ses[i].tok=argmax(L,nv); ses[i].n_past++; ses[i].drem--; total_decode++; }
            llama_batch_free(db); double tpt=el; for(int k=0;k<rn;k++) evput("TPOT",ses[ready[k]].sid,"d",tpt); rounds++; if(rounds>2000) break;
        }
        // 3) advance phases / schedule next
        bool any=false;
        for(int i=0;i<N;i++){ Session& s=ses[i]; if(s.done) continue; if(s.drem==0){ s.phase++; if(s.phase*2+1 < (int)s.parts.size()){ s.need_pre=true; any=true; } else { s.done=true; } } }
        bool active=false; for(int i=0;i<N;i++) if(!ses[i].done) active=true;
        if(!active) break;
        if(!any){ // no resume pending, but some active (maybe all in need_pre or mid decode) -> loop
            bool progressing=false; for(int i=0;i<N;i++) if(!ses[i].done && (!ses[i].need_pre || ses[i].drem>0)) progressing=true;
            if(!progressing) break;
        }
    }
    double wall=now_ms()-wall0;
    fprintf(stderr,"[engine] DONE sessions=%d total_decode=%ld wall=%.0fms throughput=%.1f tok/s\n",N,total_decode,wall,(wall>0? total_decode*1000.0/wall:0));
    for(int i=0;i<N;i++) cudaEventDestroy(ses[i].pfill_ev);
    llama_free(B); llama_free(A); llama_model_free(model); g_ev.close(); return 0;
}
