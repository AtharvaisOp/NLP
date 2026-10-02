"""Held-out classification metrics shared by training and evaluation."""

from __future__ import annotations

from typing import Iterable


LABEL_NAMES = ["negative", "neutral", "positive"]


def classification_metrics(
    gold: Iterable[int], predicted: Iterable[int], probabilities: Iterable[Iterable[float]] | None = None
) -> dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    gold_values = list(gold)
    predicted_values = list(predicted)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_values,
        predicted_values,
        labels=[0, 1, 2],
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        gold_values,
        predicted_values,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(gold_values, predicted_values)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(weighted[2]),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(LABEL_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            gold_values, predicted_values, labels=[0, 1, 2]
        ).tolist(),
    }


def prediction_records(
    records: Iterable[object],
    predicted: Iterable[int],
    probabilities: Iterable[Iterable[float]],
) -> list[dict[str, object]]:
    rows = []
    for record, prediction, probability in zip(records, predicted, probabilities):
        probability_list = [float(value) for value in probability]
        rows.append(
            {
                "id": record.record_id,
                "text": record.text,
                "gold_label": record.label_name,
                "predicted_label": LABEL_NAMES[int(prediction)],
                "confidence": max(probability_list),
                "class_probabilities": {
                    name: probability_list[index] for index, name in enumerate(LABEL_NAMES)
                },
            }
        )
    return rows
