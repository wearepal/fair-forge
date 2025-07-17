import numpy as np
from sklearn import config_context
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import fair_forge as ff


def test_upsampler():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    groups = np.array([0, 1, 0, 1, 1], dtype=np.int32)
    with config_context(enable_metadata_routing=True):
        lr = LogisticRegression(random_state=42, max_iter=10)
        upsampler = ff.Upsampler(strategy=ff.UpsampleStrategy.NAIVE, random_state=42)
        upsampler.set_fit_request(groups=True)
        # upsampler.set_transform_request(y=True, groups=True)
        upsampler.set_transform_request(groups=True)
        pipeline = Pipeline([("upsampler", upsampler), ("classifier", lr)])
        pipeline.fit(X, y, groups=groups)
        pipeline.predict(X)
    assert False
