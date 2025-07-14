from fair_forge import methods
from sklearn.linear_model import LogisticRegression
import numpy as np


def test_reweighting():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    groups = np.array([0, 1, 0, 1, 1], dtype=np.int32)

    random = np.random.RandomState(42)
    lr = LogisticRegression(random_state=random, max_iter=10)
    method = methods.Reweighting(lr)
    method.fit(X, y, groups=groups)
    predictions = method.predict(X)
    np.testing.assert_allclose(predictions, np.array([0, 0, 1, 1, 1], dtype=np.int32))


def test_majority():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 0, 0], dtype=np.int32)

    method = methods.Majority()
    method.fit(X, y)
    predictions = method.predict(X)
    np.testing.assert_allclose(predictions, np.array([0, 0, 0, 0, 0], dtype=np.int32))


def test_blind():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 0, 0], dtype=np.int32)

    method = methods.Blind(seed=1)
    method.fit(X, y)
    predictions = method.predict(X)
    np.testing.assert_allclose(predictions, np.array([0, 1, 1, 0, 1], dtype=np.int32))
