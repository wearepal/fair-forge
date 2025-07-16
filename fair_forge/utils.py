import numpy as np


def reproducible_random_state(seed: int) -> np.random.Generator:
    """Create a random state that is reproducible across Python versions and platforms."""
    # MT19937 isn't the best random number generator, but it's reproducible, so we're using it.
    return np.random.Generator(np.random.MT19937(seed))
