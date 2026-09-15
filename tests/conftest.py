from __future__ import annotations

import pytest

from insurabench.data.datasets import make_synthetic_two_table


@pytest.fixture
def simple_book():
    """A synthetic book with one term per policy -- bare policy_id is unique."""
    return make_synthetic_two_table(n_policies=50, renewals=False, seed=1)


@pytest.fixture
def renewal_book():
    """A synthetic book with renewal terms -- policy_id recurs, term_start
    disambiguates."""
    return make_synthetic_two_table(n_policies=50, renewals=True, seed=2)
