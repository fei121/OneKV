// A-only decode on the REAL cold trace (native single-context). Compare TPOT.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
extern "C" void ggml_cuda_as_set_stream(int n);
static std::mutex g;
static double now(){ using namespace std::chrono; return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count()/1e3; }
static int argmax(const float*L,int n){ int b=0; for(int i=1;i<n;i++) if(L[i]>L[b]) b=i; return b; }
static std::string b64d(const std::string& in){ static const std::string T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; std::string out; int val=0,bits=0; for(char c:in){ if(c=='=')break; auto p=T.find(c); if(p==std::string::npos)continue; val=(val<<6)|p; bits+=6; if(bits>=8){ bits-=8; out+=(char)((val>>bits)&0xff);} } return out; }
int main(){
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=99;
    llama_model* m=llama_model_load_from_file("/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf",mp);
    llama_context_params cpa=llama_context_default_params(); cpa.n_ctx=8192; cpa.n_batch=4096; cpa.n_seq_max=1;
    llama_context* A=llama_init_from_model(m,cpa);
    std::string line; std::ifstream ifs("/root/autodl-tmp/exp/traces/sessions.txt"); std::getline(ifs,line);
    std::vector<std::string> parts; std::stringstream ss(line); std::string p; while(std::getline(ss,p,'|')) parts.push_back(p);
    std::string cold=b64d(parts[1]); int d0=atoi(parts[2].c_str());
    int nv=llama_vocab_n_tokens(llama_model_get_vocab(m)); std::vector<llama_token> toks(20000);
    int nt=llama_tokenize(llama_model_get_vocab(m),cold.c_str(),(int)cold.size(),toks.data(),(int)toks.size(),true,true);
    llama_batch pb=llama_batch_init(nt,0,1);
    for(int i=0;i<nt;i++){ pb.token[i]=toks[i]; pb.pos[i]=i; pb.n_seq_id[i]=1; pb.seq_id[i][0]=0; pb.logits[i]=(i==nt-1);} pb.n_tokens=nt;
    {std::lock_guard<std::mutex> lk(g); ggml_cuda_as_set_stream(0);} double t0=now(); int rc=llama_decode(A,pb); double ttft=now()-t0;
    const float* LA=llama_get_logits(A); int tok=argmax(LA,nv);
    fprintf(stderr,"[verifyA] A-only cold n=%d rc=%d TTFT=%.1fms first=%d\n",nt,rc,ttft,tok);
    int pos=nt; std::vector<double> tg; llama_batch db=llama_batch_init(1,0,1);
    for(int i=0;i<d0;i++){ db.token[0]=tok; db.pos[0]=pos; db.n_seq_id[0]=1; db.seq_id[0][0]=0; db.logits[0]=1; db.n_tokens=1;
        {std::lock_guard<std::mutex> lk(g); ggml_cuda_as_set_stream(0);} double a=now(); int r=llama_decode(A,db); cudaDeviceSynchronize(); double b=now(); tg.push_back(b-a); pos++;
        const float* L=llama_get_logits(A); int ntok=argmax(L,nv);
        if(i<6) fprintf(stderr,"[verifyA] step%d rc=%d tok=%d time=%.2fms\n",i,r,tok,b-a);
        tok=ntok; }
    double s=0; for(double x:tg) s+=x; fprintf(stderr,"[verifyA] TPOT=%.2fms over %d steps\n",s/d0,d0);
    llama_batch_free(db); llama_free(A); llama_model_free(m); return 0;
}
