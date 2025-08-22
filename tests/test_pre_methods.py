import numpy as np
from sklearn.linear_model import LogisticRegression

import fair_forge as ff


def test_upsampler():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    groups = np.array([0, 1, 0, 1, 1], dtype=np.int32)
    lr = LogisticRegression(random_state=41, max_iter=10)
    upsampler = ff.Upsampler(strategy="naive", random_state=41)
    pipeline = ff.GroupPipeline(group_data_modifier=upsampler, estimator=lr)
    pipeline.set_params(random_state=42)
    assert pipeline.get_params()["estimator__random_state"] == 42
    assert pipeline.get_params()["group_data_modifier__random_state"] == 42
    pipeline.fit(X, y, groups=groups)
    predictions = pipeline.predict(X)
    np.testing.assert_allclose(predictions, np.array([0, 0, 1, 1, 1], dtype=np.int32))
