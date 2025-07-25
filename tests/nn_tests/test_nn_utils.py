from itertools import islice

import jax.numpy as jnp

from fair_forge import nn as ffn


def test_iterate_forever() -> None:
    """Test the iterate_forever function."""
    data = (jnp.arange(10), jnp.arange(10, 20))
    batch_size = 3
    seed = 42
    num_batches = 10
    batches = list(
        islice(ffn.iterate_forever(data, batch_size=batch_size, seed=seed), num_batches)
    )

    # Check that we have the expected number of batches.
    assert len(batches) == num_batches

    # Check that each batch has the correct shape.
    for batch in batches:
        assert all(d.shape[0] == batch_size for d in batch), str(batches)

    # Check that the data is shuffled.
    assert not jnp.array_equal(batches[0][0], data[0][:batch_size])
