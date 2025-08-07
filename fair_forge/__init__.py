"""
.. image:: _static/fair_forge_logo.svg

Example usage:

>>> import fair_forge as ff
>>> import numpy as np
>>> from sklearn.linear_model import LogisticRegression
>>> X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
>>> y = np.array([0, 0, 1, 1, 1], dtype=np.int32)
>>> groups = np.array([0, 1, 0, 1, 1], dtype=np.int32)
>>> lr = LogisticRegression(random_state=42, max_iter=10)
>>> method = ff.Reweighting(lr)
>>> method.fit(X, y, groups=groups)
Reweighting(base_method=LogisticRegression(max_iter=10, random_state=42))
>>> method.predict(X)
array([0, 0, 1, 1, 1], dtype=int32)
"""

from .datasets import *
from .eval import *
from .methods import *
from .metrics import *
from .preprocessing import *
from .split import *
from .utils import *
