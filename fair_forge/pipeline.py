from collections.abc import Sequence
from enum import Enum
from typing import cast

import polars as pl

from fair_forge.datasets import GroupDataset
from fair_forge.methods import GroupMethod, Method
from fair_forge.metrics import Float, GroupMetric, Metric
from fair_forge.preprocessing import Preprocessor
from fair_forge.split import basic_split, proportional_split
from fair_forge.utils import reproducible_random_state

__all__ = ["Split", "evaluate"]


class Split(Enum):
    BASIC = "basic"
    PROPORTIONAL = "proportional"


def evaluate(
    dataset: GroupDataset,
    methods: Sequence[Method | GroupMethod],
    metrics: Sequence[Metric],
    group_metrics: Sequence[GroupMetric],
    *,
    preprocessor: Preprocessor | None = None,
    repeat: int = 1,
    split: Split = Split.PROPORTIONAL,
    seed: int = 42,
    train_percentage: float = 0.8,
    remove_score_suffix: bool = True,
) -> dict[str, pl.DataFrame]:
    result: dict[int, dict[str, dict[str, Float]]] = {}

    generator = reproducible_random_state(seed)

    for repeat_index in range(repeat):
        if repeat_index not in result:
            result[repeat_index] = {}
        match split:
            case Split.BASIC:
                train_idx, test_idx = basic_split(
                    generator,
                    train_percentage,
                    target=dataset.target,
                    groups=dataset.groups,
                )
            case Split.PROPORTIONAL:
                train_idx, test_idx = proportional_split(
                    generator,
                    train_percentage,
                    target=dataset.target,
                    groups=dataset.groups,
                )
        train_x = dataset.data[train_idx]
        train_y = dataset.target[train_idx]
        train_groups = dataset.groups[train_idx]
        test_x = dataset.data[test_idx]
        test_y = dataset.target[test_idx]
        test_groups = dataset.groups[test_idx]

        if preprocessor is not None:
            train_x = preprocessor.fit(train_x).transform(train_x)
            test_x = preprocessor.transform(test_x)

        for method in methods:
            method_name: str = repr(method)
            if method_name not in result[repeat_index]:
                result[repeat_index][method_name] = {}

            # If a method requests `groups` in its metadata, we cast it to GroupMethod.
            if "groups" in method.get_metadata_routing().fit.requests:  # type: ignore
                cast(GroupMethod, method).fit(
                    train_x,
                    train_y,
                    groups=train_groups,
                )
            else:
                cast(Method, method).fit(train_x, train_y)

            predictions = method.predict(test_x)

            for metric in metrics:
                metric_name = metric.__name__
                if remove_score_suffix and metric_name.endswith("_score"):
                    metric_name = metric_name[:-6]
                score = metric(y_true=test_y, y_pred=predictions)
                result[repeat_index][method_name][metric_name] = score

            for group_metric in group_metrics:
                group_metric_name = group_metric.__name__
                group_score = group_metric(
                    y_true=test_y, y_pred=predictions, groups=test_groups
                )
                result[repeat_index][method_name][group_metric_name] = group_score

    # Convert the result dictionary to a Polars DataFrame
    # We have to perform a kind of "transpose" operation here
    method_names = list(result[0])
    r: dict[str, pl.DataFrame] = {}
    for method_name in method_names:
        df = pl.DataFrame(
            [result[repeat_index][method_name] for repeat_index in range(repeat)]
        )
        r[method_name] = df
    return r
