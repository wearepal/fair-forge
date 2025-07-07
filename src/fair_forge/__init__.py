from typing import Protocol
import numpy as np
from numpy.typing import NDArray
from sklearn import set_config
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate


def my_score(
    y_true: NDArray[np.float32],
    y_pred: NDArray[np.float32],
    groups: NDArray[np.float32],
) -> float:
    assert groups is not None, "Groups must be provided for scoring"
    return 1.0


class Method(Protocol):
    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
        groups: NDArray[np.float32],
    ): ...

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]: ...


class ExampleClassifier(ClassifierMixin, BaseEstimator):
    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
        groups: NDArray[np.float32],
    ):
        assert groups is not None, "Groups must be provided for fitting"
        # all classifiers need to expose a classes_ attribute once they're fit.
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        # return a constant value of 1, not a very smart classifier!
        return np.ones(len(X))


if __name__ == "__main__":
    set_config(enable_metadata_routing=True)
    n_samples, n_features = 100, 4

    rng = np.random.RandomState(42)
    X = rng.rand(n_samples, n_features)
    y = rng.randint(0, 2, size=n_samples)
    my_groups = rng.randint(0, 10, size=n_samples)

    scaler = StandardScaler()
    # scaler.set_fit_request(sample_weights=False)
    classifier = ExampleClassifier()
    classifier.set_fit_request(groups=True)

    pipe = Pipeline(
        [
            ("scaler", scaler),
            ("classifier", classifier),
        ]
    )

    pipe.fit(X, y, groups=my_groups)

    pipe.predict(X)

    scorer = make_scorer(
        my_score,
        greater_is_better=True,
    )
    scorer.set_score_request(groups=True)

    cross_validate(
        pipe,
        X,
        y,
        scoring=scorer,
        cv=5,
        return_train_score=True,
        params={"groups": my_groups},
    )
