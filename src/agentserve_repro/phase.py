#!/usr/bin/env python3
"""Task 10 — Request phase classification (ColdPrefill / ResumePrefill / Decode).

Follows plan's rules:
  ColdPrefill   : no reusable KV prefix AND new input tokens
  ResumePrefill : session KV exists AND new input tokens
  Decode        : KV established AND generating next token(s), no new input
"""
from dataclasses import dataclass, field
from enum import Enum

class RequestPhase(Enum):
    COLD_PREFILL = "cold_prefill"
    RESUME_PREFILL = "resume_prefill"
    DECODE = "decode"

@dataclass
class AgentRequestMeta:
    session_id: int
    request_id: int
    phase: RequestPhase
    total_context_tokens: int
    cached_prefix_tokens: int
    new_input_tokens: int
    requested_output_tokens: int
    kv_prefix_available: bool
    arrival_ts_ns: int

class RequestClassifier:
    @staticmethod
    def classify(meta: AgentRequestMeta) -> RequestPhase:
        if meta.new_input_tokens > 0 and not meta.kv_prefix_available:
            return RequestPhase.COLD_PREFILL
        if meta.new_input_tokens > 0 and meta.kv_prefix_available:
            return RequestPhase.RESUME_PREFILL
        if meta.new_input_tokens == 0 and meta.requested_output_tokens > 0:
            return RequestPhase.DECODE
        raise ValueError(f"cannot classify request: {meta}")

def build_metas_from_session(session_plan, session_id=0):
    """Yield AgentRequestMeta for each request in a trace plan.

    Each plan entry = (prefill + decode). We emit one prefill request meta and one
    decode request meta. KV availability = whether the session has processed prior
    context (i.e. not the first phase).
    """
    metas = []
    rid = 0
    for i, phase in enumerate(session_plan):
        p = phase["phase"]
        inp = phase["input_tokens"]
        out = phase["decode_tokens"]
        has_kv = i > 0
        rid += 1
        # prefill request (cold or resume)
        metas.append(AgentRequestMeta(
            session_id=session_id, request_id=rid,
            phase=RequestPhase.COLD_PREFILL if not has_kv else RequestPhase.RESUME_PREFILL,
            total_context_tokens=inp if not has_kv else (phase["input_tokens"]),
            cached_prefix_tokens=0 if not has_kv else 0,
            new_input_tokens=inp, requested_output_tokens=0,
            kv_prefix_available=has_kv, arrival_ts_ns=0))
        # decode request (generates next tokens on established KV)
        rid += 1
        metas.append(AgentRequestMeta(
            session_id=session_id, request_id=rid, phase=RequestPhase.DECODE,
            total_context_tokens=inp, cached_prefix_tokens=inp,
            new_input_tokens=0, requested_output_tokens=out,
            kv_prefix_available=True, arrival_ts_ns=0))
    return metas

def self_test():
    # Acceptance: initial->Cold, tool append->Resume, next-token->Decode
    c = RequestClassifier.classify
    cold = AgentRequestMeta(1,1,None,3000,0,3000,0,False,0)
    res  = AgentRequestMeta(1,2,None,3056,3000,56,0,True,0)
    dec  = AgentRequestMeta(1,3,None,3056,3056,0,38,True,0)
    assert c(cold) == RequestPhase.COLD_PREFILL, c(cold)
    assert c(res)  == RequestPhase.RESUME_PREFILL, c(res)
    assert c(dec)  == RequestPhase.DECODE, c(dec)
    print("ACCEPTANCE OK: initial->ColdPrefill, tool-append->ResumePrefill, next-token->Decode")
    return True

if __name__ == "__main__":
    import json, sys
    # run acceptance test by default
    self_test()
    # optional: classify a trace file's plan
    if len(sys.argv) > 1:
        lines=[json.loads(l) for l in open(sys.argv[1])]
        from collections import Counter
        cnt=Counter()
        for s in lines:
            for m in build_metas_from_session(s["plan"]):
                cnt[RequestClassifier.classify(m).value]+=1
        print("trace classification counts:", dict(cnt))
