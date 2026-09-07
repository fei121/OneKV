import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agentserve_repro.phase import RequestPhase, AgentRequestMeta, RequestClassifier, self_test

def test_classify():
    assert RequestClassifier.classify(AgentRequestMeta(1,1,None,3000,0,3000,0,False,0))==RequestPhase.COLD_PREFILL
    assert RequestClassifier.classify(AgentRequestMeta(1,2,None,3056,3000,56,0,True,0))==RequestPhase.RESUME_PREFILL
    assert RequestClassifier.classify(AgentRequestMeta(1,3,None,3056,3056,0,38,True,0))==RequestPhase.DECODE
