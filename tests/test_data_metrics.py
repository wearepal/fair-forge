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
                "y_proportions_ratio": 0.3295113,
                "group_proportions_ratio": 2.077373,
                "combination_proportions_ratio": 12.57519,
                "y_proportions_min": 0.2478439,
                "group_proportions_min": 0.324952,
                "combination_proportions_min": 0.0369068,
                "y_proportions_max": 0.7521560,
                "group_proportions_max": 0.675047,
                "combination_proportions_max": 0.464110,
                "hgr_corr": 0.215760,
            }
        ),
        check_column_order=False,
    )
