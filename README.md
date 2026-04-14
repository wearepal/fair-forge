![logo](./docs/source/_static/fair_forge_logo.svg)

## What is fair-forge?

fair-forge is a Python toolkit for developing and evaluating **group-aware machine learning** methods — models that take sensitive attributes (such as sex or race) into account during training in order to produce fairer outcomes.

The library provides:

- **Fairness metrics**: group-aware metrics that measure disparities across demographic groups (e.g. equalized odds gaps, demographic parity ratios). Standard metrics like accuracy can be used directly from [scikit-learn](https://scikit-learn.org), and fair-forge can turn any such metric into a group-aware variant via `as_group_metric()`.
- **Fairness-aware methods**: these come in two flavours: *preprocessing methods* like `Upsampler`, which modify the training data and can be combined with any scikit-learn estimator via `GroupPipeline`; and *in-processing methods* like `Reweighting` ([Kamiran & Calders, 2012](https://link.springer.com/article/10.1007/s10115-011-0463-8)), which alter the training dynamics directly (e.g. by adjusting sample weights).
- **An evaluation pipeline**: a single `evaluate()` call that handles train/test splitting, preprocessing, training multiple methods, and computing all metrics, returning a tidy Polars DataFrame of results.
- **Example datasets**: including the Adult income dataset, ready to use out of the box.

fair-forge is designed around **[scikit-learn](https://scikit-learn.org) compatibility**: methods follow the estimator API, standard scikit-learn metrics and models can be used directly, and preprocessing transforms work with scikit-learn Pipelines via metadata routing. This means you can mix and match fair-forge components with the broader scikit-learn ecosystem without any adapter code.

### Quick example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import fair_forge as ff

# Load the Adult dataset with "Sex" as the sensitive attribute
adult = ff.load_adult("Sex")

# Compare a standard model against a fairness-aware variant
methods = {
    "LR": LogisticRegression(),
    "Reweighting": ff.Reweighting(LogisticRegression()),
}

# Evaluate with accuracy and the minimum accuracy over groups (“robust accuracy”)
results = ff.evaluate(
    dataset=adult,
    methods=methods,
    metrics=[accuracy_score],
    group_metrics=ff.as_group_metric((accuracy_score,), ff.MetricAgg.MIN),
)
print(results)
```

For a complete walkthrough, see [`examples/full_example.ipynb`](examples/full_example.ipynb).

## Installation
This library requires at least Python 3.12. Install it from pypi:

```sh
pip install fair-forge
```

or from GitHub:

```sh
pip install git+https://github.com/wearepal/fair-forge.git
```

If you want to use the neural-network-based methods, you need to add the `nn` extras:

```sh
pip install 'fair-forge[nn]'
```

or

```sh
pip install 'fair-forge[nn] @ git+https://github.com/wearepal/fair-forge.git'
```

## Usage
`fair-forge` provides two main components: metrics and methods. Besides these, there are various utility functions to help with common tasks and also a few example datasets.

The core data type used in `forge-fair` is **numpy arrays**: all the methods and metrics expect numpy arrays as input. If you have data in a different form, it is usually easy to convert it to numpy arrays:

- Pandas: [to_numpy()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html)
- Polars: [to_numpy()](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_numpy.html)
- PyTorch: [numpy()](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)
- TensorFlow: [make_ndarray()](https://www.tensorflow.org/api_docs/python/tf/make_ndarray)

### Metrics
There are group-aware metrics and non-group-aware metrics. The non-group-aware metrics are callables with this function signature:

```python
import numpy as np
from numpy.typing import NDArray

type Float = float | np.float16 | np.float32 | np.float64

def tpr(y_true: NDArray[np.int32], y_pred: NDArray[np.int32]) -> Float: ...
```

In other words, a non-group-aware metric accepts two numpy arrays — one with the true labels and one with the predicted labels — and returns a single `Float`. The API of the non-group-aware metrics is chosen such that any metric from scikit-learn can be used — for example, [accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).

Group-aware metrics take an additional parameter, the group labels:

```python
def cv(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    *,
    groups: NDArray[np.int32],
) -> Float:
```

A very important function is `fair_forge.as_group_metric()`. It takes in a non-group-aware metric, and turns it into one or more group-aware metrics. This is done by first computing the metric value per group, and these individual metric values are then aggregated in different ways — for example, by taking the minimum or the ratio of the values. Here is how one would construct a *robust accuracy* metric (minimum accuracy across all groups):

```python
import fair_forge as ff
from sklearn.metrics import accuracy_score

# Construct a metric for the minimum accuracy over all groups
(robust_accuracy,) = ff.as_group_metric(
    (accuracy_score,), agg=ff.MetricAgg.MIN
)

# Use it as a group-aware metric
robust_accuracy(y_true=y_true, y_pred=y_pred, groups=groups)
```

### Methods
The group-aware vs non-group-aware distinction also exists for the methods provided in this library. The non-group-aware methods simply follow the scikit-learn API for an estimator (inheriting from `BaseEstimator` adds some mixin methods which are needed):

```python
from sklearn.base import BaseEstimator

class Method(BaseEstimator):
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32]) -> Self:
        pass

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        pass
```

The methods can be used like normal scikit-learn estimators.

On the other hand, we have the group-based methods, which take an additional parameter, the group labels:

```python
from sklearn.base import BaseEstimator

class GroupMethod(BaseEstimator):
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int32], *, group: NDArray[np.int32]) -> Self:
        pass

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        pass
```

These methods can use the group information at training time to produce fairer models.

Besides methods which output a machine learning model, there are also methods which transform the data. These then have a `transform` method instead of a `predict` method:

```python
from sklearn.base import BaseEstimator

class MyGroupBasedTransform(BaseEstimator):
    def fit(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> Self:
        pass

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        pass

    def fit_transform(
        self, X: NDArray[np.float32], y: NDArray[np.int32], *, groups: NDArray[np.int32]
    ) -> NDArray[np.float32]:
        pass
```

(Unfortunately, you have to implement `fit_transform` manually, because otherwise it will not pass on the `groups` parameter.)

Such transformation methods can then be combined with non-group-aware methods with scikit-learn’s [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html):

```python
from sklearn import config_context
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Pipeline will only forward the `groups` argument if we
# set `enable_metadata_routing` to `True`.
with config_context(enable_metadata_routing=True):
    estimator = LinearSVC(random_state=42, max_iter=100)
    transform = MyGroupBasedTransform(random_state=42)
    # We need to explicitly request here that the transformation's
    # `fit` method gets the `groups` argument.
    transform.set_fit_request(groups=True)

    pipeline = Pipeline([("transform", transform), ("estimator", estimator)])

    # This will call `fit_and_transform` on MyGroupBasedTransform
    pipeline.fit(train_x, train_y, groups=train_groups)
    preds = pipeline.predict(test_x)
```

Alternatively, you can use `ff.GroupPipeline` which handles this specific case directly, without the need for metadata routing. However, `ff.GroupPipeline` is limited to exactly one transformation and one estimator.

### Utilities
`fair-forge` provides many useful components for running experiments and collecting results:

- example datasets (like Adult)
- train-test splitting
- facilities for running multiple methods and evaluating them with multiple metrics

For more information on this, see the documentation.
