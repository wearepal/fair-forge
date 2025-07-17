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

    lr_results = result[0]
    assert lr_results.method_name == "LogisticRegression"
    # TODO: Only check a subset of the parameters.
    assert lr_results.params == {
        "C": 1.0,
        "class_weight": None,
        "dual": False,
        "fit_intercept": True,
        "intercept_scaling": 1,
        "l1_ratio": None,
        "max_iter": 10,
        "multi_class": "deprecated",
        "n_jobs": None,
        "penalty": "l2",
        "random_state": 42,
        "solver": "lbfgs",
        "tol": 0.0001,
        "verbose": 0,
        "warm_start": False,
    }

    assert_frame_equal(
        lr_results.scores,
        pl.DataFrame(
            {
                "repeat_index": [0, 1],
                "split_seed": [42, 43],
                "accuracy": [0.8, 0.9],
                "prob_pos_min": [0.46153846153846156, 0.38461538461538464],
                "prob_pos_max": [0.7142857142857143, 0.5714285714285714],
            }
        ),
    )

    assert result[1].method_name == "Blind"
