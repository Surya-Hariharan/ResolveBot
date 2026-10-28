import pytest

from app.confirmation import is_clear_yes


@pytest.mark.parametrize("message", [
    "yes", "Yes", "YES!", "yes.", "Yes, go ahead.", "go ahead", "confirm", "Confirmed", "please do",
    "yes please", "yep", "yeah", "do it", "yes do it", "Yes, please go ahead", "proceed", "absolutely",
    "ok go ahead", "yes create it", "yes thanks",
])
def test_clear_yes(message):
    assert is_clear_yes(message)


@pytest.mark.parametrize("message", [
    "", "   ", "no", "No thanks", "nope", "maybe", "ok", "okay", "sure", "hmm", "I think so", "not yet",
    "yes but only the charger", "yes, just the power bank", "yes... actually no", "wait", "stop",
    "cancel it", "don't", "yes no", "what will happen?", "can you explain first", "just do it",
    "yes and also cancel my other order", "no, go ahead and forget it",
    "sure I guess so maybe but I am not certain about this at all really",
])
def test_not_a_clear_yes(message):
    assert not is_clear_yes(message)
