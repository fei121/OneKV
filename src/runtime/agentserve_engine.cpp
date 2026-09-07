// Single-engine shared-KV serving engine (AgentServe reproduction).
//
// Two llama_contexts share one KV pool (ctx_other=A): A=prefill, B=decode. P/D on separate CUDA
// streams synchronized by cudaEvent; a mutex protects the shared cell bookkeeping.
// Continuous batching: one llama_decode(B) packs one token per ready session.
//
// Fairness fixes (so the engine is measured on the SAME workload as the llama.cpp / vLLM / SGLang
// baselines):
//   * Sequence reuse: serve a FIXED set of MAXSESS (default 12) sessions over N concurrent slots.
//     seq 0 is a permanent template holding the shared system prefix; slots use seq ids 1..N and
//     are recycled when a session finishes (llama_memory_seq_rm + seq_cp + unique prefill).
//   * EOS respect: after each decode, if the argmax token is an end-of-generation token the phase
//     ends immediately (no forced generation to the token cap).
//   * tool_wait: after each step (except the last) the session sleeps tool_wait_ms before its next
//     resume prefill, matching the harness's simulated external tool-call latency.
//
// Trace (per session): sid|cold_b64|d0|app1_b64|d1|app2_b64|d2|...  ; phase i (i>=1) input =
// parts[1+2i], decode count = parts[2+2i]; phase 0 cold = parts[1], decode = parts[2].
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <fstream>
#include <sstream>
#include <deque>
#include <chrono>
#include <cuda_runtime.h>
extern "C" void ggml_cuda_as_set_stream(int n);

static std::mutex g_kv; static std::mutex g_io; static std::ofstream g_ev;
static double now_ms(){ using namespace std::chrono; return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count()/1e3; }
static int argmax(const float*L,int n){ int b=0; for(int i=1;i<n;i++) if(L[i]>L[b]) b=i; return b; }
static std::string b64d(const std::string& in){ static const std::string T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; std::string out; int val=0,bits=0; for(char c:in){ if(c=='=')break; auto p=T.find(c); if(p==std::string::npos)continue; val=(val<<6)|p; bits+=6; if(bits>=8){ bits-=8; out+=(char)((val>>bits)&0xff);} } return out; }

struct Session {
    std::string sid; std::vector<std::string> parts;
    int seq=0; int phase=0; int n_past=0; int tok=0; int drem=0;
    int need_pre=0; bool done=false; bool active=false;
    cudaEvent_t pfill_ev=nullptr; double wait_until=0.0; double start_ms=0.0;
};
struct RawSession { std::string sid; std::vector<std::string> parts; };

static llama_context* A; static llama_context* B; static const llama_vocab* v; static int nv;
static std::vector<llama_token> toks(16384); static int pre_stream, dec_stream;
static std::vector<Session> ses;          // N slots, seq ids 1..N
static std::vector<RawSession> allp;      // MAXSESS loaded sessions
static std::deque<int> q;                 // pending session indices (into allp)
static int g_common=0; static int g_N=0;
static long total_decode=0;

static void evput(const std::string&e,const std::string&sid,const char*ph,double ms){ std::lock_guard<std::mutex> lk(g_io); g_ev<<"{\"event\":\""<<e<<"\",\"session\":\""<<sid<<"\",\"phase\":\""<<ph<<"\",\"ms\":"<<ms<<"}\n"; g_ev.flush(); }

// Activate pending session `pidx` into slot `slot` (seq id = slot+1): reset the seq, copy the
// shared system prefix from template seq 0, prefill the session's unique instruction, set the
// first decode token and the first phase decode budget.
static void activate(int slot, int pidx, double tool_wait_ms){
    Session& s = ses[slot];
    s.sid = allp[pidx].sid; s.parts = allp[pidx].parts;
    s.seq = slot + 1; s.phase = 0; s.done = false; s.need_pre = 0; s.wait_until = 0.0;
    s.start_ms = now_ms();
    int sidx = s.seq; int common = g_common;
    std::string cold = b64d(s.parts[1]);
    int nt = llama_tokenize(v, (const char*)cold.c_str(), (int)cold.size(), toks.data(), (int)toks.size(), true, true);
    llama_memory_t memA = llama_get_memory(A);
    { std::lock_guard<std::mutex> lk(g_kv);
        ggml_cuda_as_set_stream(pre_stream);
        llama_memory_seq_rm(memA, sidx, 0, -1);
        llama_memory_seq_cp(memA, 0, sidx, 0, common);  // copy FULL shared prefix (p1 exclusive)
    }
    int nuni = nt - common; int start = (int)(llama_memory_seq_pos_max(memA, sidx) + 1);
    double a = now_ms();
    if(nuni>0){ llama_batch ub = llama_batch_init(nuni,0,1);
        for(int k=0;k<nuni;k++){ ub.token[k]=toks[common+k]; ub.pos[k]=start+k; ub.n_seq_id[k]=1; ub.seq_id[k][0]=sidx; ub.logits[k]=(k==nuni-1); } ub.n_tokens=nuni;
        { std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(pre_stream); llama_decode(A,ub); cudaDeviceSynchronize(); } llama_batch_free(ub);
    }
    double cold_ttft = now_ms() - a;   // per-session cold prefill (prefix cached)
    { std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(pre_stream); cudaDeviceSynchronize(); }
    const float* L = llama_get_logits(A);
    s.tok = argmax(L, nv); s.n_past = (int)(llama_memory_seq_pos_max(memA, sidx) + 1);
    s.drem = atoi(s.parts[2].c_str());
    s.active = true; s.need_pre = 0;
    evput("TTFT_cold", s.sid, "c", cold_ttft);
    (void)tool_wait_ms;
}

int main(int argc,char**argv){
    std::string trace=argc>1?argv[1]:"/root/autodl-tmp/exp/traces/sessions.txt";
    int N=argc>2?atoi(argv[2]):3;
    pre_stream=argc>3?atoi(argv[3]):-1; dec_stream=argc>4?atoi(argv[4]):1;
    int NCX=argc>5?atoi(argv[5]):49152;
    const char* MOD=argc>6?argv[6]:"/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf";
    int MAXSESS=argc>7?atoi(argv[7]):12;
    int tool_wait_ms=argc>8?atoi(argv[8]):200;
    g_N = N;
    g_ev.open("/root/autodl-tmp/exp/raw_logs/as_batch_events.jsonl");

    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=99;
    llama_model* model=llama_model_load_from_file(MOD,mp);
    llama_context_params cpa=llama_context_default_params(); cpa.n_ctx=NCX; cpa.n_batch=20000; cpa.n_threads=4; cpa.n_seq_max=N+1; cpa.kv_unified=true; cpa.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    A=llama_init_from_model(model,cpa);
    llama_context_params cpb=llama_context_default_params(); cpb.n_ctx=NCX; cpb.n_batch=20000; cpb.n_threads=4; cpb.n_seq_max=N+1; cpb.ctx_other=A; cpb.kv_unified=true; cpb.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    B=llama_init_from_model(model,cpb);
    v=llama_model_get_vocab(model); nv=llama_vocab_n_tokens(v);

    // Load up to MAXSESS sessions from the trace (same set the baselines drive).
    { std::ifstream ifs(trace); std::string line; while((int)allp.size()<MAXSESS && std::getline(ifs,line)){ if(line.empty())continue;
        std::stringstream ss(line); std::string p; RawSession rs; while(std::getline(ss,p,'|')) rs.parts.push_back(p);
        rs.sid=rs.parts[0]; allp.push_back(rs); } }
    int NSESS=(int)allp.size(); if(NSESS==0){ fprintf(stderr,"[engine] no sessions\n"); return 1; }
    int nsess_used = std::min(NSESS, MAXSESS);
    for(int i=0;i<nsess_used;i++) q.push_back(i);

    // Find the common system-prefix length across all loaded sessions (prefix caching).
    { std::vector<std::vector<llama_token>> stoks(nsess_used); int minlen=INT32_MAX;
      for(int i=0;i<nsess_used;i++){ std::string cold=b64d(allp[i].parts[1]); int nt=llama_tokenize(v,(const char*)cold.c_str(),(int)cold.size(),toks.data(),(int)toks.size(),true,true);
          stoks[i].assign(toks.begin(),toks.begin()+nt); minlen=std::min(minlen,nt); }
      int common=0; for(int k=0;k<minlen;k++){ bool same=true; for(int i=1;i<nsess_used;i++) if(stoks[i][k]!=stoks[0][k]){same=false;break;} if(same) common++; else break; }
      g_common=common;
      // Prefill the shared system prefix once on the template seq 0.
      llama_batch pb=llama_batch_init(common,0,1);
      for(int k=0;k<common;k++){ pb.token[k]=stoks[0][k]; pb.pos[k]=k; pb.n_seq_id[k]=1; pb.seq_id[k][0]=0; pb.logits[k]=(k==common-1);} pb.n_tokens=common;
      { std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(pre_stream); llama_decode(A,pb); cudaDeviceSynchronize(); } llama_batch_free(pb);
      fprintf(stderr,"[engine] common_prefill_tokens=%d sessions=%d\n",common,nsess_used);
    }

    ses.resize(N);
    for(int s=0;s<N;s++) cudaEventCreateWithFlags(&ses[s].pfill_ev, cudaEventDisableTiming);
    double wall0=now_ms();

    // Activate the first N sessions into slots.
    for(int slot=0; slot<N && !q.empty(); slot++){ int pidx=q.front(); q.pop_front(); activate(slot,pidx,(double)tool_wait_ms); }

    int guard=0;
    while(guard<200000){
        guard++;
        // 1) resume prefills for sessions past their tool_wait door
        bool any_pre=false;
        for(int s=0;s<N;s++) if(ses[s].active && !ses[s].done && ses[s].need_pre && now_ms()>=ses[s].wait_until) any_pre=true;
        if(any_pre){
            for(int s=0;s<N;s++){ Session& S=ses[s]; if(!S.active||S.done||!S.need_pre||now_ms()<S.wait_until) continue;
                int ph=S.phase; std::string text=b64d(S.parts[1+2*ph]);
                // Match the baseline's resume context: the tool result is wrapped in <tool_result>...</tool_result>
                // (verified == jsonl phase[i].prompt - phase[i-1].prompt for every phase/session).
                text = "\n\n<tool_result>\n" + text + "\n</tool_result>\n";
                int nt=llama_tokenize(v,(const char*)text.c_str(),(int)text.size(),toks.data(),(int)toks.size(),false,true);
                llama_batch pb=llama_batch_init(nt,0,1);
                for(int k=0;k<nt;k++){ pb.token[k]=toks[k]; pb.pos[k]=S.n_past+k; pb.n_seq_id[k]=1; pb.seq_id[k][0]=S.seq; pb.logits[k]=(k==nt-1);} pb.n_tokens=nt;
                double t0=now_ms(); { std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(pre_stream); llama_decode(A,pb); if(S.pfill_ev) cudaEventRecord(S.pfill_ev); }
                cudaDeviceSynchronize(); double el=now_ms()-t0; llama_batch_free(pb);
                S.n_past+=nt; const float* L=llama_get_logits(A); S.tok=argmax(L,nv); S.drem=atoi(S.parts[2+2*ph].c_str()); S.need_pre=0;
                evput("TTFT_resume",S.sid,"r",el);
                fprintf(stderr,"[engine] %s phase%d resume TTFT=%.1fms drem=%d\n",S.sid.c_str(),ph,el,S.drem);
            }
        }
        // 2) batched decode rounds
        int rounds=0;
        while(true){
            std::vector<int> ready; for(int s=0;s<N;s++) if(ses[s].active && !ses[s].done && ses[s].drem>0) ready.push_back(s);
            if(ready.empty()) break;
            int rn=(int)ready.size(); llama_batch db=llama_batch_init(rn,0,N+1);
            for(int k=0;k<rn;k++){ int s=ready[k]; db.token[k]=ses[s].tok; db.pos[k]=ses[s].n_past; db.n_seq_id[k]=1; db.seq_id[k][0]=ses[s].seq; db.logits[k]=1; } db.n_tokens=rn;
            double a,b; { std::lock_guard<std::mutex> lk(g_kv); ggml_cuda_as_set_stream(dec_stream); a=now_ms(); llama_decode(B,db); } cudaDeviceSynchronize(); b=now_ms(); double el=b-a;
            for(int k=0;k<rn;k++){ int s=ready[k]; const float* L=llama_get_logits_ith(B,k); llama_token t=argmax(L,nv);
                if(llama_vocab_is_eog(v,t)){ ses[s].drem=0; ses[s].n_past++; }   // EOS: end phase; still advance past the fed token
                else { ses[s].tok=t; ses[s].n_past++; ses[s].drem--; total_decode++; }
            }
            llama_batch_free(db);
            for(int k=0;k<rn;k++) evput("TPOT",ses[ready[k]].sid,"d",el);
            rounds++; if(rounds>2000) break;
        }
        // 3) advance phases / recycle slots
        for(int s=0;s<N;s++){ Session& S=ses[s]; if(!S.active||S.done) continue;
            if(S.drem==0 && S.need_pre==0){   // only advance once the phase's decode truly finished
                S.phase++;
                if(S.phase*2+1 < (int)S.parts.size()){ S.need_pre=1; S.wait_until=now_ms()+tool_wait_ms; }  // tool_wait before next step
                else { S.done=true; S.active=false;
                    if(!q.empty()){ int pidx=q.front(); q.pop_front(); activate(s,pidx,(double)tool_wait_ms); } }
            }
        }
        // termination: only when NO active session remains AND the queue is empty.
        // (Do NOT break when active sessions are merely waiting on a tool_wait -- the loop must
        //  keep going until the wait elapses and their resume prefill / decode runs.)
        bool any_active=false;
        for(int s=0;s<N;s++) if(ses[s].active && !ses[s].done) any_active=true;
        bool any_pending = !q.empty();
        if(!any_active && !any_pending) break;
        // avoid busy-spin: if the only active sessions are past their tool_wait door (nothing
        // immediately actionable), sleep briefly so the next deadline arrives.
        bool any_ready=false;
        for(int s=0;s<N;s++){ Session& S=ses[s]; if(S.active && !S.done){
            if(S.drem>0 || (S.need_pre && now_ms()>=S.wait_until)){ any_ready=true; break; } } }
        if(!any_ready) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    double wall=now_ms()-wall0;
    int completed=0; for(int s=0;s<N;s++) if(ses[s].done) completed++;
    fprintf(stderr,"[engine] DONE slots=%d sessions_run=%d total_decode=%ld wall=%.0fms throughput=%.1f tok/s\n",N,nsess_used,total_decode,wall,(wall>0? total_decode*1000.0/wall:0));
    for(int s=0;s<N;s++) cudaEventDestroy(ses[s].pfill_ev);
    llama_free(B); llama_free(A); llama_model_free(model); g_ev.close(); return 0;
}
