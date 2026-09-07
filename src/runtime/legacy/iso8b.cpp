#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <mutex>
extern "C" void ggml_cuda_as_set_stream(int n);
static std::mutex g;
static int argmax(const float*L,int n){ int b=0; for(int i=1;i<n;i++) if(L[i]>L[b]) b=i; return b; }
int main(){
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=99;
    llama_model* m=llama_model_load_from_file("/root/autodl-tmp/models/Qwen2.5-3B-f16.gguf",mp);
    const char* p="Hello, what is the weather in Sydney today?";
    std::vector<llama_token> toks(4096); int nv=llama_vocab_n_tokens(llama_model_get_vocab(m));
    auto gen=[&](int mode,int ngen){
        llama_context_params cpa=llama_context_default_params(); cpa.n_ctx=4096; cpa.n_batch=4096; cpa.n_seq_max=1;
        llama_context* A=llama_init_from_model(m,cpa);
        llama_context* dec=nullptr;
        if(mode==1){ llama_context_params cpb=llama_context_default_params(); cpb.n_ctx=4096; cpb.n_batch=4096; cpb.n_seq_max=1; cpb.ctx_other=A; dec=llama_init_from_model(m,cpb); }
        else { dec=A; }
        int n=llama_tokenize(llama_model_get_vocab(m),p,strlen(p),toks.data(),4096,true,true);
        llama_batch pb=llama_batch_init(n,0,1);
        for(int i=0;i<n;i++){ pb.token[i]=toks[i]; pb.pos[i]=i; pb.n_seq_id[i]=1; pb.seq_id[i][0]=0; pb.logits[i]=(i==n-1);} pb.n_tokens=n;
        { std::lock_guard<std::mutex> lk(g); ggml_cuda_as_set_stream(0);} llama_decode(A,pb);
        // first token comes from A's prefill logits
        const float* LA=llama_get_logits(A); int tok=argmax(LA,nv);
        std::vector<int> seq; seq.push_back(tok);
        int pos=n; llama_batch db=llama_batch_init(1,0,1);
        for(int gg=0;gg<ngen;gg++){   // produce ngen more tokens
            db.token[0]=tok; db.pos[0]=pos; db.n_seq_id[0]=1; db.seq_id[0][0]=0; db.logits[0]=1; db.n_tokens=1;
            {std::lock_guard<std::mutex> lk(g); ggml_cuda_as_set_stream(0);} llama_decode(dec,db); pos++;
            const float* L=llama_get_logits(dec); tok=argmax(L,nv); seq.push_back(tok);
        }
        llama_batch_free(db); if(mode==1) llama_free(dec); llama_free(A);
        return seq;
    };
    auto s0=gen(0,8); auto s1=gen(1,8);
    fprintf(stderr,"[iso8] single(ctx A) : "); for(int x:s0) fprintf(stderr,"%d ",x); fprintf(stderr,"\n");
    fprintf(stderr,"[iso8] shared(A->B)  : "); for(int x:s1) fprintf(stderr,"%d ",x); fprintf(stderr,"\n");
    fprintf(stderr,"[iso8] MATCH=%s\n",(s0==s1)?"YES":"NO");
    llama_model_free(m);
    return (s0==s1)?0:1;
}
