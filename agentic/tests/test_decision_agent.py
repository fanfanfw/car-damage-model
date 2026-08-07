from agentic.decision_agent import DecisionAgent, make_decision
from agentic.schemas import DamageSignal


def test_auto_approve_minor_scratch():
    agent = DecisionAgent(policies_dir="policies")
    signal = DamageSignal(damage_type="scratch", confidence=0.92, severity="minor")
    decision = agent.decide(signal)
    assert decision.action == "AUTO_APPROVE"


def test_make_decision_matches_ui_for_light_scratch():
    result = make_decision([{"type": "scratch", "severity": "light", "confidence": 0.9}])
    assert result["decision"] == "AUTO_APPROVE"


def test_make_decision_matches_ui_for_multiple_damages():
    damages = [
        {"type": "scratch", "severity": "light", "confidence": 0.9},
        {"type": "dent", "severity": "moderate", "confidence": 0.7},
    ]
    result = make_decision(damages)
    assert result["decision"] == "HUMAN_REVIEW"


def test_low_confidence_human_review():
    agent = DecisionAgent(policies_dir="policies")
    signal = DamageSignal(damage_type="dent", confidence=0.40, severity="moderate")
    decision = agent.decide(signal)
    assert decision.action == "HUMAN_REVIEW"
