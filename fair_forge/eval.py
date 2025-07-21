from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, cast

import polars as pl

from fair_forge.datasets import GroupDataset
from fair_forge.methods import GroupMethod, Method
from fair_forge.metrics import GroupMetric, Metric
from fair_forge.preprocessing import Preprocessor
from fair_forge.split import SplitMethod, basic_split, proportional_split

__all__ = ["Split", "evaluate"]


class Split(Enum):
    BASIC = "basic"
    PROPORTIONAL = "proportional"


def evaluate(
    dataset: GroupDataset,
    methods: Mapping[str, Method | GroupMethod],
    metrics: Sequence[Metric],
    group_metrics: Sequence[GroupMetric],
    *,
    preprocessor: Preprocessor | None = None,
    repeat: int = 1,
    split: Split | SplitMethod = Split.PROPORTIONAL,
    seed: int = 42,
    train_percentage: float = 0.8,
    remove_score_suffix: bool = True,
    seed_methods: bool = True,
) -> pl.DataFrame:
    result: list[dict[str, Any]] = []

    for repeat_index in range(repeat):
        split_seed = seed + repeat_index
        split_method: SplitMethod
        match split:
            case Split.BASIC:
                split_method = basic_split
            case Split.PROPORTIONAL:
                split_method = proportional_split
            case _:
                split_method = split
        train_idx, test_idx = split_method(
            split_seed, train_percentage, target=dataset.target, groups=dataset.groups
        )

        train_x = dataset.data[train_idx]
        train_y = dataset.target[train_idx]
        train_groups = dataset.groups[train_idx]
        test_x = dataset.data[test_idx]
        test_y = dataset.target[test_idx]
        test_groups = dataset.groups[test_idx]

        if preprocessor is not None:
            train_x = preprocessor.fit(train_x, train_y).transform(train_x)
            test_x = preprocessor.transform(test_x)

        for method_name, method in methods.items():
            row: dict[str, Any] = {}
            row["method"] = method_name
            row["repeat_index"] = repeat_index
            row["split_seed"] = split_seed

            if seed_methods and "random_state" in method.get_params():
                # If the method has a `random_state` parameter, we set it.
                method.set_params(random_state=split_seed)

            # If a method requests `groups` in its metadata, we cast it to GroupMethod.
            if "groups" in method.get_metadata_routing().fit.requests:
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
                row[metric_name] = score

            for group_metric in group_metrics:
                group_metric_name = group_metric.__name__
                group_score = group_metric(
                    y_true=test_y, y_pred=predictions, groups=test_groups
                )
                row[group_metric_name] = group_score
            result.append(row)

    # Convert the result list to a Polars DataFrame.
    # We use `pl.Enum` to ensure the correct ordering of method names.
    method_names = pl.Enum(list(methods))
    return pl.DataFrame(
        result,
        schema_overrides={
            "method": method_names,
            "repeat_index": pl.Int64,
            "split_seed": pl.Int64,
        },
    ).sort("method", "repeat_index")
