from __future__ import annotations

import pytest

from backend.app.services.demo_sentiment import demo_sentiment


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("हे उत्पादन खूप चांगले आहे!", "positive"),
        ("हा मोबाईल खूप चांगला आहे.", "positive"),
        ("चांगली सेवा", "positive"),
        ("ही सेवा अत्यंत खराब आहे.", "negative"),
        ("आज दुकान सकाळी दहा वाजता उघडले.", "neutral"),
        ("हा phone चांगला आहे पण battery backup खराब आहे.", "negative"),
        ("हा product चांगला नाही!!!", "negative"),
        ("सेवा खराब नाही", "positive"),
        ("सेवा चांगली अजिबात नाही", "negative"),
        ("चांगले नाही आणि खराब आहे", "negative"),
        ("चांगले नाही आणि वाईट आहे", "negative"),
        ("चांगले।", "positive"),
        ("खराब।", "negative"),
        ("चांगले॥", "positive"),
        ("चांगले। खराब आहे", "neutral"),
        ("चांगले नाही। खराब आहे", "negative"),
        ("This phone is GOOD!", "positive"),
        ("This is not a good phone", "negative"),
        ("This is not bad", "positive"),
        ("I don't like it", "negative"),
        ("Good, but not great", "negative"),
        ("चांगले आणि खराब", "neutral"),
        ("Good. Not bad.", "positive"),
        ("Not. Good.", "positive"),
        ("Goodbye notebook badminton", "neutral"),
        ("काम नाही", "neutral"),
        ("unknown words 📰", "neutral"),
    ],
)
def test_demo_is_input_dependent_with_bounded_normalized_scores(text: str, expected: str) -> None:
    result = demo_sentiment(text)
    assert result.label == expected
    assert set(result.probabilities) == {"negative", "neutral", "positive"}
    assert sum(result.probabilities.values()) == pytest.approx(1)
    assert result.confidence == result.probabilities[result.label]
    assert all(0 < score < 1 for score in result.probabilities.values())
    assert result.confidence <= 0.8


def test_repetition_cannot_manufacture_certain_demo_scores() -> None:
    assert demo_sentiment("good " * 1000) == demo_sentiment("good good good")
