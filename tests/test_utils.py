import fair_forge as ff


def test_reproducible_random_state() -> None:
    """Test the reproducible_random_state function."""
    seed = 42
    rng1 = ff.reproducible_random_state(seed)
    rng2 = ff.reproducible_random_state(seed)
    # Check that the random states are equal.
    assert rng1 is not rng2  # Different instances
    assert rng1.integers(100) == rng2.integers(100)  # Same output for the same seed


def test_batched() -> None:
    """Test the batched function."""
    len_data = 10
    batch_size = 3
    batches = list(ff.batched(len_data, batch_size, drop_last=False))
    # Check that we have the expected number of batches.
    expected_num_batches = (len_data + batch_size - 1) // batch_size
    assert len(batches) == expected_num_batches
    # Check that each batch has the correct start and end indices.
    for i, batch in enumerate(batches):
        start = i * batch_size
        end = min(start + batch_size, len_data)
        assert batch.start == start
        assert batch.stop == end
    # Test with drop_last=True
    drop_last_batches = list(ff.batched(len_data, batch_size, drop_last=True))
    expected_num_batches_drop_last = len_data // batch_size
    assert len(drop_last_batches) == expected_num_batches_drop_last
    # Check that each batch has the correct start and end indices.
    for i, batch in enumerate(drop_last_batches):
        start = i * batch_size
        end = start + batch_size
        assert batch.start == start
        assert batch.stop == end
