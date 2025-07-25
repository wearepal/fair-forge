import numpy as np
from sklearn import config_context
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import fair_forge as ff
from fair_forge import nn as ffn


def test_beutel_standalone():
    toy = ff.load_ethicml_toy()
    train_idx, test_idx = ff.basic_split(
        seed=0, train_percentage=0.8, target=toy.target, groups=toy.groups
    )
    train_x = toy.data[train_idx]
    train_y = toy.target[train_idx]
    train_groups = toy.groups[train_idx]
    test_x = toy.data[test_idx]

    enc_size = 2
    beutel = ffn.Beutel(enc_size=[enc_size], random_state=42)
    beutel.fit(train_x, train_y, groups=train_groups)
    transformed = beutel.transform(test_x)
    assert transformed.shape == (test_x.shape[0], enc_size)


def test_beutel_pipeline():
    toy = ff.load_ethicml_toy()
    train_idx, test_idx = ff.basic_split(
        seed=0, train_percentage=0.8, target=toy.target, groups=toy.groups
    )
    train_x = toy.data[train_idx]
    train_y = toy.target[train_idx]
    train_groups = toy.groups[train_idx]
    test_x = toy.data[test_idx]

    with config_context(enable_metadata_routing=True):
        estimator = LinearSVC(random_state=42, max_iter=100)
        transform = ffn.Beutel(enc_size=[2], random_state=42)
        transform.set_fit_request(groups=True)

        pipeline = Pipeline([("transform", transform), ("estimator", estimator)])

        pipeline.fit(train_x, train_y, groups=train_groups)
        preds = pipeline.predict(test_x)

    assert np.count_nonzero(preds) == 49
