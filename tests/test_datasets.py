from fair_forge import datasets
import numpy as np


def test_adult_gender():
    data = datasets.adult_dataset(
        group=datasets.AdultGroup.SEX,
        group_in_features=False,
        binarize_nationality=False,
        binarize_race=False,
    )

    assert data.name == "Adult Sex"
    assert data.data.shape == (45222, 101)
    assert data.target.shape == (45222,)
    np.testing.assert_allclose(np.unique(data.target), np.array([0, 1], dtype=np.int32))
    assert data.groups.shape == (45222,)
    np.testing.assert_allclose(np.unique(data.groups), np.array([0, 1], dtype=np.int32))
    assert data.feature_grouping[0] == slice(1, 8)  # workclass
    assert data.feature_grouping[1] == slice(8, 24)  # education
    assert data.feature_grouping[2] == slice(25, 32)  # marital-status
    assert data.feature_grouping[3] == slice(32, 46)  # occupation
    assert data.feature_grouping[4] == slice(46, 52)  # relationship
    assert data.feature_grouping[5] == slice(52, 57)  # race
    assert data.feature_grouping[6] == slice(60, 101)  # native-country


def test_adult_race():
    data = datasets.adult_dataset(
        group=datasets.AdultGroup.RACE,
        group_in_features=False,
        binarize_nationality=False,
        binarize_race=False,
    )

    assert data.name == "Adult Race"
    assert data.data.shape == (45222, 98)
    assert data.target.shape == (45222,)
    assert data.groups.shape == (45222,)
    np.testing.assert_allclose(
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
    data = datasets.adult_dataset(
        group=datasets.AdultGroup.RACE,
        group_in_features=False,
        binarize_nationality=True,
        binarize_race=True,
    )

    assert data.name == "Adult Race, binary nationality, binary race"
    assert data.data.shape == (45222, 59)
    assert data.target.shape == (45222,)
    assert data.groups.shape == (45222,)
    np.testing.assert_allclose(np.unique(data.groups), np.array([0, 1], dtype=np.int32))
    assert data.feature_grouping[0] == slice(1, 8)  # workclass
    assert data.feature_grouping[1] == slice(8, 24)  # education
    assert data.feature_grouping[2] == slice(25, 32)  # marital-status
    assert data.feature_grouping[3] == slice(32, 46)  # occupation
    assert data.feature_grouping[4] == slice(46, 52)  # relationship
    assert data.feature_grouping[5] == slice(52, 54)  # sex
    assert data.feature_grouping[6] == slice(57, 59)  # native-country


def test_adult_gender_in_features():
    data = datasets.adult_dataset(
        group=datasets.AdultGroup.SEX,
        group_in_features=True,
        binarize_nationality=True,
        binarize_race=False,
    )

    assert data.name == "Adult Sex, binary nationality, group in features"
    assert data.data.shape == (45222, 64)
    assert data.target.shape == (45222,)
    assert data.groups.shape == (45222,)
    np.testing.assert_allclose(np.unique(data.groups), np.array([0, 1], dtype=np.int32))
    assert data.feature_grouping[0] == slice(1, 8)  # workclass
    assert data.feature_grouping[1] == slice(8, 24)  # education
    assert data.feature_grouping[2] == slice(25, 32)  # marital-status
    assert data.feature_grouping[3] == slice(32, 46)  # occupation
    assert data.feature_grouping[4] == slice(46, 52)  # relationship
    assert data.feature_grouping[5] == slice(52, 57)  # race
    assert data.feature_grouping[6] == slice(57, 59)  # sex
    assert data.feature_grouping[7] == slice(62, 64)  # native-country
