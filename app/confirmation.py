"""Decides whether a message is a clear "yes". Plain Python, never the LLM.

A message counts as a yes only when every word is a confirming word or harmless
filler, and at least one word really confirms. So "yes, go ahead" and "please do"
pass, while "yes but only the charger", "ok", "maybe" and "no" do not.
"""
import re

CONFIRMING_WORDS = {
    "yes", "yeah", "yep", "yup", "y", "confirm", "confirmed", "proceed", "ahead",
    "absolutely", "definitely", "affirmative",
}
# "please do" and "do it" confirm even though "do" and "it" are ordinary words
CONFIRMING_PHRASES = ("please do", "do it", "do so")
FILLER_WORDS = {
    "please", "pls", "thanks", "thank", "you", "kindly", "go", "ahead", "and", "now", "then",
    "sure", "ok", "okay", "do", "it", "so", "that", "is", "right", "correct", "the", "request",
    "create", "submit", "file", "raise", "open", "for", "me", "i", "confirm", "yes",
}


def is_clear_yes(message):
    text = re.sub(r"[^a-z\s]", " ", message.lower())
    words = text.split()
    if not words or len(words) > 8:
        return False
    if any(word not in FILLER_WORDS and word not in CONFIRMING_WORDS for word in words):
        return False
    normalized = " ".join(words)
    return any(word in CONFIRMING_WORDS for word in words) or any(
        phrase in normalized for phrase in CONFIRMING_PHRASES
    )
