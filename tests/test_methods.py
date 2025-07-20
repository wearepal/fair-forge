import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import fair_forge as ff


def test_reweighting():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    groups = np.array([0, 1, 0, 1, 1], dtype=np.int32)

    lr = LogisticRegression(random_state=42, max_iter=10)
    method = ff.Reweighting(lr)
    method.fit(X, y, groups=groups)
    predictions = method.predict(X)
    np.testing.assert_allclose(predictions, np.array([0, 0, 1, 1, 1], dtype=np.int32))


def test_majority():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 0, 0], dtype=np.int32)

    method = ff.Majority()
    method.fit(X, y)
    predictions = method.predict(X)
    np.testing.assert_allclose(predictions, np.array([0, 0, 0, 0, 0], dtype=np.int32))


def test_blind():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 0, 0], dtype=np.int32)

    method = ff.Blind(random_state=1)
    method.fit(X, y)
    predictions = method.predict(X)
    np.testing.assert_allclose(predictions, np.array([0, 1, 1, 0, 1], dtype=np.int32))
    assert method.get_params() == {"random_state": 1}
    method.set_params(random_state=2)
    assert method.get_params() == {"random_state": 2}
    with pytest.raises(ValueError, match="Invalid parameter 'random_stat'"):
        method.set_params(random_stat=2)
