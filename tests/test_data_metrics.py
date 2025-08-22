import polars as pl
from polars.testing import assert_frame_equal

import fair_forge as ff


def test_adult_overview():
    data = ff.load_adult("Sex")
    assert_frame_equal(
        ff.data_overview(data),
        pl.DataFrame(
            {
                "y_distance_from_uniform": 0.133193,
                "group_distance_from_uniform": 0.0626009,
                "combination_distance_from_uniform": 0.221487,
                "y_imbalance_diff": -0.50431206,
                "group_imbalance_diff": 0.350095,
                "y_imbalance_ratio": 0.3295113,
                "group_imbalance_ratio": 2.077373,
                "hgr_corr": 0.215760,
            }
        ),
        check_column_order=False,
    )
