import numpy as np

import fair_forge as ff


def test_adult_gender():
    data = ff.load_adult(
        group=ff.AdultGroup.SEX,
        group_in_features=False,
        binarize_nationality=False,
        binarize_race=False,
    )

    assert data.name == "Adult Sex"
    assert data.data.shape == (45222, 101)
    assert data.target.shape == (45222,)
    np.testing.assert_array_equal(
        np.unique(data.target), np.array([0, 1], dtype=np.int32)
    )
    assert data.groups.shape == (45222,)
    np.testing.assert_array_equal(
        np.unique(data.groups), np.array([0, 1], dtype=np.int32)
    )
    assert data.feature_grouping[0] == slice(1, 8)  # workclass
    assert data.feature_grouping[1] == slice(8, 24)  # education
    assert data.feature_grouping[2] == slice(25, 32)  # marital-status
    assert data.feature_grouping[3] == slice(32, 46)  # occupation
    assert data.feature_grouping[4] == slice(46, 52)  # relationship
    assert data.feature_grouping[5] == slice(52, 57)  # race
    assert data.feature_grouping[6] == slice(60, 101)  # native-country


def test_adult_race():
    data = ff.load_adult(
        group=ff.AdultGroup.RACE,
        group_in_features=False,
        binarize_nationality=False,
        binarize_race=False,
    )

    assert data.name == "Adult Race"
    assert data.data.shape == (45222, 98)
    assert data.target.shape == (45222,)
    assert data.groups.shape == (45222,)
    np.testing.assert_array_equal(
        np.unique(data.groups), np.array([0, 1, 2, 3, 4], dtype=np.int32)
    )
    assert data.feature_grouping[0] == slice(1, 8)  # workclass
    assert data.feature_grouping[1] == slice(8, 24)  # education
    assert data.feature_grouping[2] == slice(25, 32)  # marital-status
    assert data.feature_grouping[3] == slice(32, 46)  # occupation
    assert data.feature_grouping[4] == slice(46, 52)  # relationship
    assert data.feature_grouping[5] == slice(52, 54)  # sex
    assert data.feature_grouping[6] == slice(57, 98)  # native-country


def test_adult_race_binary():
    data = ff.load_adult(
        group=ff.AdultGroup.RACE,
        group_in_features=False,
        binarize_nationality=True,
        binarize_race=True,
    )

    assert data.name == "Adult Race, binary nationality, binary race"
    assert data.data.shape == (45222, 59)
    assert data.target.shape == (45222,)
    assert data.groups.shape == (45222,)
    np.testing.assert_array_equal(
        np.unique(data.groups), np.array([0, 1], dtype=np.int32)
    )
    assert data.feature_grouping[0] == slice(1, 8)  # workclass
    assert data.feature_grouping[1] == slice(8, 24)  # education
    assert data.feature_grouping[2] == slice(25, 32)  # marital-status
    assert data.feature_grouping[3] == slice(32, 46)  # occupation
    assert data.feature_grouping[4] == slice(46, 52)  # relationship
    assert data.feature_grouping[5] == slice(52, 54)  # sex
    assert data.feature_grouping[6] == slice(57, 59)  # native-country


def test_adult_gender_in_features():
    data = ff.load_adult(
        group=ff.AdultGroup.SEX,
        group_in_features=True,
        binarize_nationality=True,
        binarize_race=False,
    )

    assert data.name == "Adult Sex, binary nationality, group in features"
    assert data.data.shape == (45222, 64)
    assert data.target.shape == (45222,)
    assert data.groups.shape == (45222,)
    np.testing.assert_array_equal(
        np.unique(data.groups), np.array([0, 1], dtype=np.int32)
    )
    assert data.feature_grouping[0] == slice(1, 8)  # workclass
    assert data.feature_grouping[1] == slice(8, 24)  # education
    assert data.feature_grouping[2] == slice(25, 32)  # marital-status
    assert data.feature_grouping[3] == slice(32, 46)  # occupation
    assert data.feature_grouping[4] == slice(46, 52)  # relationship
    assert data.feature_grouping[5] == slice(52, 57)  # race
    assert data.feature_grouping[6] == slice(57, 59)  # sex
    assert data.feature_grouping[7] == slice(62, 64)  # native-country


def test_toy():
    data = ff.load_ethicml_toy()
    assert data.name == "Toy"
    assert data.data.shape == (400, 10)
    assert data.target.shape == (400,)
    assert data.target.sum() == 231
    assert data.groups.shape == (400,)
    assert data.groups.sum() == 200
    np.testing.assert_array_equal(
        np.unique(data.groups), np.array([0, 1], dtype=np.int32)
    )
    assert data.feature_grouping[0] == slice(2, 7)  # group 1
    assert data.feature_grouping[1] == slice(7, 10)  # group 2
