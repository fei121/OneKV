import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agentserve_repro.scheduler import PhaseRouter, AgentRequestMeta, RequestPhase

def test_route():
    r=PhaseRouter(64)
    assert r.route(AgentRequestMeta(1,1,None,3000,0,3000,0,False,0))=="QP"
    assert r.route(AgentRequestMeta(1,2,None,3056,3000,56,0,True,0))=="QD"
    assert r.route(AgentRequestMeta(1,3,None,3251,3000,251,0,True,0))=="QP"
    assert r.route(AgentRequestMeta(1,4,None,3251,3251,0,38,True,0))=="QD"
