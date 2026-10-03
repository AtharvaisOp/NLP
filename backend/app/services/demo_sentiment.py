"""Small, offline Marathi/English lexicon demo, NOT a trained classifier.

The normalized scores preserve the API contract; they are not calibrated
probabilities or accuracy claims. Unknown wording, sarcasm and complex negation
are outside this deliberately limited demo.
"""

from __future__ import annotations

import re
import unicodedata

from .interfaces import SentimentResult


DEMO_VERSION = "rule-demo-v1"
_POSITIVE = frozenset(
    "चांगला चांगली चांगले चांगल्या छान मस्त उत्तम उत्कृष्ट सुंदर आवडते आवडला "
    "आवडली आवडले आनंद आनंदी समाधानी स्वस्त अप्रतिम good great excellent nice "
    "love loved like liked amazing happy helpful awesome best".split()
)
_NEGATIVE = frozenset(
    "खराब वाईट बेकार निराश निराशाजनक भयंकर महाग त्रास त्रासदायक दुःख असमाधानी "
    "bad terrible awful poor hate hated worst disappointing disappointed sad "
    "expensive broken useless slow".split()
)
_NEGATION = frozenset(
    "नाही नको नव्हता नव्हती नव्हते नसलेला नसलेली न not never no don't doesn't "
    "isn't wasn't aren't weren't can't cannot".split()
)
_MARATHI_NEGATION = frozenset(word for word in _NEGATION if not word.isascii())
_CONTRAST = frozenset("पण परंतु मात्र but however".split())
_TOKENS = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)?|[\u0900-\u0963\u0966-\u097f]+|[.!?।॥,;:\n]+"
)


def demo_sentiment(model_text: str) -> SentimentResult:
    """Count whole-word cues with short-range negation and contrast boundaries."""
    tokens = _TOKENS.findall(unicodedata.normalize("NFC", model_text).casefold().replace("’", "'"))
    clauses: list[list[str]] = [[]]
    weights = [1]
    for token in tokens:
        if token in _CONTRAST or token[0] in ".!?।॥,;:\n":
            clauses.append([])
            weights.append(2 if token in _CONTRAST else 1)
        else:
            clauses[-1].append(token)

    counts = {"positive": 0, "negative": 0}
    for clause, weight in zip(clauses, weights):
        previous_cue = -1
        used_negations: set[int] = set()
        for index, token in enumerate(clause):
            if token not in _POSITIVE and token not in _NEGATIVE:
                continue
            polarity = "positive" if token in _POSITIVE else "negative"
            # English negation precedes a cue; Marathi can also follow it.
            negations = {
                position for position in range(max(previous_cue + 1, index - 3), index)
                if clause[position] in _NEGATION and position not in used_negations
            }
            for position in range(index + 1, min(len(clause), index + 3)):
                following = clause[position]
                if following in _POSITIVE or following in _NEGATIVE:
                    break
                if following in _MARATHI_NEGATION:
                    negations.add(position)
            if len(negations) % 2:
                polarity = "negative" if polarity == "positive" else "positive"
            used_negations.update(negations)
            counts[polarity] += weight
            previous_cue = index

    # Cap cue counts so repetition cannot create near-100% certainty. Balanced
    # conflicting cues and no recognized cues favor an explicitly uncertain neutral.
    positive, negative = (min(3, counts[name]) for name in ("positive", "negative"))
    scores = {
        "positive": 1 + 2 * positive,
        "negative": 1 + 2 * negative,
        "neutral": 1 + 2 * min(positive, negative) + (2 if positive == negative else 0),
    }
    total = sum(scores.values())
    probabilities = {name: value / total for name, value in scores.items()}
    label = max(probabilities, key=probabilities.get)
    return SentimentResult(label=label, confidence=probabilities[label], probabilities=probabilities)
