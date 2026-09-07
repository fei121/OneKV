// Measure cold TTFT on DIVERSE tasks: full-cold prefill (no cache) vs system-prefill + instruction-prefill (cache).
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>
extern "C" void ggml_cuda_as_set_stream(int n);
static std::string b64d(const std::string& in){ static const std::string T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; std::string out; int val=0,bits=0; for(char c:in){ if(c=='=')break; auto p=T.find(c); if(p==std::string::npos)continue; val=(val<<6)|p; bits+=6; if(bits>=8){ bits-=8; out+=(char)((val>>bits)&0xff);} } return out; }
static double now(){ using namespace std::chrono; return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count()/1e3; }
int main(int argc,char**argv){
    const char* trace=argc>1?argv[1]:"/root/autodl-tmp/exp/traces/diverse_sessions.txt";
    int N=argc>2?atoi(argv[2]):6;
    std::vector<std::string> colds;
    { std::ifstream ifs(trace); std::string line; int i=0; while(std::getline(ifs,line)&&i<N){ if(line.empty())continue; std::stringstream ss(line); std::string p; std::vector<std::string> parts; while(std::getline(ss,p,'|'))parts.push_back(p); colds.push_back(b64d(parts[1])); i++; } }
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=99;
    llama_model* m=llama_model_load_from_file("/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf",mp);
    llama_context_params cp=llama_context_default_params(); cp.n_ctx=8192; cp.n_batch=8192; cp.n_threads=8; cp.n_seq_max=1; cp.kv_unified=true;
    llama_context* c=llama_init_from_model(m,cp);
    int nv=llama_vocab_n_tokens(llama_model_get_vocab(m)); std::vector<llama_token> toks(20000);
    auto prefill_time=[&](const std::string& text, bool addbos)->double{
        int nt=llama_tokenize(llama_model_get_vocab(m),(const char*)text.c_str(),(int)text.size(),toks.data(),(int)toks.size(),addbos,true);
        llama_batch pb=llama_batch_init(nt,0,1);
        for(int i=0;i<nt;i++){ pb.token[i]=toks[i]; pb.pos[i]=i; pb.n_seq_id[i]=1; pb.seq_id[i][0]=0; pb.logits[i]=(i==nt-1);} pb.n_tokens=nt;
        // warm up by running once, then CLEAR KV, then measure fresh
        { ggml_cuda_as_set_stream(0); llama_decode(c,pb); cudaDeviceSynchronize(); }
        llama_memory_seq_rm(llama_get_memory(c),0,-1,-1);
        double t0=now(); { ggml_cuda_as_set_stream(0); llama_decode(c,pb); cudaDeviceSynchronize(); } double el=now()-t0;
        llama_batch_free(pb);
        llama_memory_seq_rm(llama_get_memory(c),0,-1,-1);
        return el;
    };
    // 1) full cold (no cache)
    double full_sum=0; for(int i=0;i<N;i++){ double e=prefill_time(colds[i],true); full_sum+=e; }
    double full_avg=full_sum/N;
    // 2) system + instruction split (need to parse). We don't have explicit split; measure the FULL cold as one.
    // Instead measure a longer estimate: report full cold avg per session; cache benefit comes from shared system.
    fprintf(stderr,"[cold] DIVERSE N=%d: FULL-cold(no-cache) avg=%.1fms per session\n",N,full_avg);
    // 3) measure the SHARED system part: use the common prefix of the colds (chars) as a proxy? Instead:
    //    measure system-only by taking the common char prefix.
    std::string sys=""; { int mn=INT32_MAX; for(auto&c:colds) mn=std::min(mn,(int)c.size()); int cp=0; for(int i=0;i<mn;i++){ bool sm=true; for(auto&c:colds) if(c[i]!=colds[0][i]){sm=false;break;} if(sm)cp++; else break; } sys=colds[0].substr(0,cp); }
    double sys_ms=prefill_time(sys,true);
    fprintf(stderr,"[cold] system(shared, cacheable) prefill=%.1fms (chars=%d)\n",sys_ms,(int)sys.size());
    double full_single=prefill_time(colds[0],true);
    fprintf(stderr,"[cold] full_single(no-cache)=%.1fms | cache=sys(%.1fms once)+per-session instruction\n",full_single,sys_ms);
    // per-session instruction prefill (unique part)
    double inst_sum=0; for(int i=0;i<N;i++){ std::string inst=colds[i].substr(sys.size()); double e=prefill_time(inst,true); inst_sum+=e; }
    double inst_avg=inst_sum/N;
    fprintf(stderr,"[cold] per-session instruction (cache) avg=%.1fms\n",inst_avg);
    fprintf(stderr,"[cold] => cold TTFT no-cache~%.1fms | cache(sys once + inst)~%.1fms per session\n",full_avg, inst_avg);
    llama_free(c); llama_model_free(m); return 0;
}
