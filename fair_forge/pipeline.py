from collections.abc import Sequence
from enum import Enum
from typing import NamedTuple, cast

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


class Result(NamedTuple):
    method_name: str
    params: dict[str, object]
    scores: pl.DataFrame


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
) -> list[Result]:
    result: list[list[tuple[str, dict[str, object], dict[str, Float]]]] = []

    for repeat_index in range(repeat):
        split_seed = seed + repeat_index
        generator = reproducible_random_state(split_seed)
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

        result_for_repeat: list[tuple[str, dict[str, object], dict[str, Float]]] = []
        for method in methods:
            method_name: str = method.__class__.__name__
            scores: dict[str, Float] = {}
            scores["repeat_index"] = repeat_index
            scores["split_seed"] = split_seed

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
                scores[metric_name] = score

            for group_metric in group_metrics:
                group_metric_name = group_metric.__name__
                group_score = group_metric(
                    y_true=test_y, y_pred=predictions, groups=test_groups
                )
                scores[group_metric_name] = group_score
            result_for_repeat.append((method_name, method.get_params(), scores))
        result.append(result_for_repeat)

    # Convert the result dictionary to a Polars DataFrame
    # We have to perform a kind of "transpose" operation here
    names_and_params = [(i, entry[0], entry[1]) for i, entry in enumerate(result[0])]
    r: list[Result] = []
    for i, method_name, params in names_and_params:
        df = pl.DataFrame(
            [result[repeat_index][i][2] for repeat_index in range(repeat)]
        )
        r.append(
            Result(
                method_name=method_name,
                params=params,
                scores=df,
            )
        )
    return r
