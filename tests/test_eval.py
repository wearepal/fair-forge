import polars as pl
from polars.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import fair_forge as ff


def test_pipeline_with_dummy():
    ds = ff.load_dummy_dataset(seed=42)
    lr = LogisticRegression(random_state=42, max_iter=10)
    blind = ff.Blind(random_state=42)
    group_metrics = ff.as_group_metric(
        [ff.prob_pos], agg=ff.MetricAgg.MIN_MAX, remove_score_suffix=True
    )
    metrics = [accuracy_score]

    result = ff.evaluate(
        dataset=ds,
        methods={"LR": lr, "Blind": blind},
        metrics=metrics,
        group_metrics=group_metrics,
        repeat=2,
        split="basic",
        seed=42,
        train_percentage=0.8,
        remove_score_suffix=True,
        preprocessor=StandardScaler(),
        seed_methods=False,
    )

    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "method": ["LR", "LR", "Blind", "Blind"],
                "repeat_index": [0, 1, 0, 1],
                "split_seed": [42, 43, 42, 43],
                "accuracy": [0.8, 0.9, 0.6, 0.6],
                "prob_pos_min": [
                    0.46153846153846156,
                    0.38461538461538464,
                    0.42857142857142855,
                    0.2857142857142857,
                ],
                "prob_pos_max": [
                    0.7142857142857143,
                    0.5714285714285714,
                    0.7692307692307693,
                    0.8461538461538461,
                ],
            },
            schema_overrides={"method": pl.Enum(["LR", "Blind"])},
        ),
    )
