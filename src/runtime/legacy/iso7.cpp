#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <mutex>
extern "C" void ggml_cuda_as_set_stream(int n);
static std::mutex g;
int main(){
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=99;
    llama_model* m=llama_model_load_from_file("/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf",mp);
    const char* p="Hello, what is the weather in Sydney today?";
    std::vector<llama_token> toks(4096); int nv=llama_vocab_n_tokens(llama_model_get_vocab(m));
    llama_context_params cpa=llama_context_default_params(); cpa.n_ctx=4096; cpa.n_batch=4096; cpa.n_seq_max=1;
    llama_context* A=llama_init_from_model(m,cpa);
    // create B BEFORE prefill
    llama_context_params cpb=llama_context_default_params(); cpb.n_ctx=4096; cpb.n_batch=4096; cpb.n_seq_max=1; cpb.ctx_other=A;
    llama_context* B=llama_init_from_model(m,cpb);
    int n=llama_tokenize(llama_model_get_vocab(m),p,strlen(p),toks.data(),4096,true,true);
    llama_batch pb=llama_batch_init(n,0,1);
    for(int i=0;i<n;i++){ pb.token[i]=toks[i]; pb.pos[i]=i; pb.n_seq_id[i]=1; pb.seq_id[i][0]=0; pb.logits[i]=(i==n-1); } pb.n_tokens=n;
    { std::lock_guard<std::mutex> lk(g); ggml_cuda_as_set_stream(0); }
    int rc=llama_decode(A,pb);
    llama_memory_t memA=llama_get_memory(A), memB=llama_get_memory(B);
    llama_pos amin=llama_memory_seq_pos_min(memA,0), amax=llama_memory_seq_pos_max(memA,0);
    llama_pos bmin=llama_memory_seq_pos_min(memB,0), bmax=llama_memory_seq_pos_max(memB,0);
    fprintf(stderr,"[iso7] A prefill rc=%d n=%d  A seq0=%ld..%ld  B seq0=%ld..%ld\n",rc,n,(long)amin,(long)amax,(long)bmin,(long)bmax);
    // decode on B
    int best=0; const float* LA=llama_get_logits(A); for(int i=1;i<nv;i++) if(LA[i]>LA[best]) best=i;
    llama_batch db=llama_batch_init(1,0,1); db.token[0]=best; db.pos[0]=n; db.n_seq_id[0]=1; db.seq_id[0][0]=0; db.logits[0]=1; db.n_tokens=1;
    int rcB=llama_decode(B,db); const float* LB=llama_get_logits(B);
    fprintf(stderr,"[iso7] B decode rc=%d best=%d LB[0]=%f LB[5]=%f\n",rcB,best,LB[0],LB[5]);
    llama_free(B); llama_free(A); llama_model_free(m);
    return 0;
}
