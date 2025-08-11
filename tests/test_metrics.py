import numpy as np
from sklearn.metrics import accuracy_score

import fair_forge as ff
from fair_forge.metrics import _PerSensMetricBase  # pyright: ignore


def test_renyi():
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
    y_pred = np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)
    groups = np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)

    renyi_y = ff.RenyiCorrelation(ff.DependencyTarget.Y)
    result = renyi_y(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1 / 3)
    assert renyi_y.__name__ == "renyi_y"

    renyi_s = ff.RenyiCorrelation(ff.DependencyTarget.S)
    result = renyi_s(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1.0)
    assert renyi_s.__name__ == "renyi_s"


def test_prob_pos():
    y_true = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    y_pred = np.array([1, 0, 1, 0, 0], dtype=np.int32)
    result = ff.prob_pos(y_true=y_true, y_pred=y_pred)
    np.testing.assert_allclose(result, 0.4)
    assert ff.prob_pos.__name__ == "prob_pos"


def test_prob_pos_masked():
    y_true = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    y_pred = np.array([1, 0, 1, 0, 0], dtype=np.int32)
    mask = np.array([True, False, True, True, False], dtype=np.bool)
    result = ff.prob_pos(y_true=y_true, y_pred=y_pred, sample_weight=mask)
    np.testing.assert_allclose(result, 2 / 3)
    assert ff.prob_pos.__name__ == "prob_pos"


def test_prob_neg():
    y_true = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    y_pred = np.array([1, 0, 1, 0, 0], dtype=np.int32)
    result = ff.prob_neg(y_true=y_true, y_pred=y_pred)
    np.testing.assert_allclose(result, 0.6)
    assert ff.prob_neg.__name__ == "prob_neg"


def test_per_sens_metrics():
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
    y_pred = np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)

    metric_list = ff.as_group_metric(
        base_metrics=(accuracy_score, ff.prob_pos),
        agg=ff.MetricAgg.ALL,
        remove_score_suffix=True,
    )
    assert len(metric_list) == 12

    assert len(_PerSensMetricBase.score_cache) == 0

    metric = metric_list[0]
    assert metric.__name__ == "accuracy_diff"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 2 / 3)

    metric = metric_list[1]
    assert metric.__name__ == "accuracy_ratio"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1 / 3)

    metric = metric_list[2]
    assert metric.__name__ == "accuracy_min"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1 / 3)

    metric = metric_list[3]
    assert metric.__name__ == "accuracy_max"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1.0)

    metric = metric_list[4]
    assert metric.__name__ == "accuracy_0"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1 / 3)

    metric = metric_list[5]
    assert metric.__name__ == "accuracy_1"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1.0)

    metric = metric_list[6]
    assert metric.__name__ == "prob_pos_diff"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1 / 3)

    metric = metric_list[7]
    assert metric.__name__ == "prob_pos_ratio"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1 / 2)

    metric = metric_list[8]
    assert metric.__name__ == "prob_pos_min"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1 / 3)

    metric = metric_list[9]
    assert metric.__name__ == "prob_pos_max"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 2 / 3)

    metric = metric_list[10]
    assert metric.__name__ == "prob_pos_0"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 1 / 3)

    metric = metric_list[11]
    assert metric.__name__ == "prob_pos_1"
    result = metric(y_true=y_true, y_pred=y_pred, groups=groups)
    np.testing.assert_allclose(result, 2 / 3)

    assert len(_PerSensMetricBase.score_cache) == 2
