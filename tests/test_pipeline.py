import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from polars.testing import assert_frame_equal
import polars as pl

from fair_forge import datasets, methods, metrics, pipeline


def test_pipeline_with_dummy():
    ds = datasets.load_dummy_dataset(seed=42)
    random_state = np.random.RandomState(42)
    lr = LogisticRegression(random_state=random_state, max_iter=10)
    blind = methods.Blind(seed=42)
    group_metrics = metrics.per_sens_metrics(
        [metrics.prob_pos], per_sens=metrics.PerSens.MIN_MAX, remove_score_suffix=True
    )
    other_metrics = [accuracy_score]

    result = pipeline.evaluate(
        dataset=ds,
        methods=[lr, blind],
        metrics=other_metrics,
        group_metrics=group_metrics,
        repeat=2,
        split=pipeline.Split.BASIC,
        seed=42,
        train_percentage=0.8,
        remove_score_suffix=True,
    )

    assert_frame_equal(
        result["LogisticRegression"],
        pl.DataFrame(
            {
                "accuracy": [0.8, 0.95],
                "prob_pos_min": [0.46153846153846156, 0.23076923076923078],
                "prob_pos_max": [0.7142857142857143, 0.7142857142857143],
            }
        ),
    )
