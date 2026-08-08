from src.evaluation import (
    calculate_metrics,
    create_class_distribution,
    create_confusion_matrix,
)


def test_calculate_metrics():
    y_true = [
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
    ]

    y_pred = [
        "Positive",
        "Negative",
        "Neutral",
        "Negative",
    ]

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "macro_f1" in metrics

    assert 0 <= metrics["precision_macro"] <= 1
    assert 0 <= metrics["recall_macro"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_confusion_matrix():
    y_true = [
        "Positive",
        "Negative",
        "Neutral",
    ]

    y_pred = [
        "Positive",
        "Negative",
        "Neutral",
    ]

    matrix = create_confusion_matrix(
        y_true,
        y_pred,
    )

    assert matrix.shape == (3, 3)


def test_class_distribution():
    labels = [
        "Positive",
        "Positive",
        "Negative",
        "Neutral",
    ]

    distribution = create_class_distribution(labels)

    assert distribution["Positive"] == 2
    assert distribution["Negative"] == 1
    assert distribution["Neutral"] == 1
