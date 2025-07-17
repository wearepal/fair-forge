import polars as pl
from polars.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import fair_forge as ff


def test_pipeline_with_dummy():
    ds = ff.load_dummy_dataset(seed=42)
    lr = LogisticRegression(random_state=42, max_iter=10)
    blind = ff.Blind(seed=42)
    metrics = ff.per_sens_metrics(
        [ff.prob_pos], per_sens=ff.PerSens.MIN_MAX, remove_score_suffix=True
    )
    other_metrics = [accuracy_score]

    result = ff.evaluate(
        dataset=ds,
        methods=[lr, blind],
        metrics=other_metrics,
        group_metrics=metrics,
        repeat=2,
        split=ff.Split.BASIC,
        seed=42,
        train_percentage=0.8,
        remove_score_suffix=True,
    )

    assert_frame_equal(
        result["LogisticRegression(max_iter=10, random_state=42)"],
        pl.DataFrame(
            {
                "accuracy": [0.8, 0.95],
                "prob_pos_min": [0.46153846153846156, 0.23076923076923078],
                "prob_pos_max": [0.7142857142857143, 0.7142857142857143],
            }
        ),
    )

    assert "Blind(seed=42)" in result
