"""Test permutation.py module."""
import numpy as np
import pytest

import pte_stats


class TestPermutationOnesample:
    """Class for testing permutation_onesample()."""

    @pytest.mark.parametrize("n_perm", [100, 1000])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    @pytest.mark.parametrize("two_tailed", [True, False])
    def test_all_data_is_baseline(self, n_perm, value, two_tailed) -> None:
        """Test case where all values are equal to baseline."""
        data = np.ones(shape=(10,), dtype=float) * value
        z, p = pte_stats.permutation_onesample(
            data_a=data, data_b=value, n_perm=n_perm, two_tailed=two_tailed
        )
        assert p == pytest.approx(1.0)
        assert z == pytest.approx(0.0)

    @pytest.mark.parametrize("two_tailed", [True, False])
    def test_zero_diff(self, two_tailed) -> None:
        """Test case where  values are non significant against baseline."""
        n_perm = 10000
        data = np.ones(shape=(20,), dtype=float)
        data[::2] = data[::2] * -1
        z, p = pte_stats.permutation_onesample(
            data_a=data, data_b=0.0, n_perm=n_perm, two_tailed=two_tailed
        )
        print(f"{p=}")
        print(f"{z=}")
        # assert p == pytest.approx(1.0)
        assert z == pytest.approx(0.0)

    @pytest.mark.parametrize("value", [1.0, -1.0])
    @pytest.mark.parametrize("two_tailed", [True, False])
    def test_baseline_zero(self, value, two_tailed):
        """Test against baseline of 0.0."""
        n_perm = 10000
        data = np.ones(shape=(20,), dtype=float) * value
        z, p = pte_stats.permutation_onesample(
            data_a=data, data_b=0.0, n_perm=n_perm, two_tailed=two_tailed
        )
        if not two_tailed and value == -1.0:
            assert p == pytest.approx(1.0)
            assert z == pytest.approx(value)
            return
        assert p == pytest.approx(1 / (n_perm + 1))
        assert z == pytest.approx(abs(value))


class TestPermutationTwosample:
    """Class for testing permutation_twosample."""

    @pytest.mark.parametrize("value", [1.0, -1.0])
    @pytest.mark.parametrize("two_tailed", [True, False])
    def test_baseline_zero(self, value, two_tailed):
        """Test against baseline of 0.0."""
        n_perm = 10000
        data_a = np.ones(shape=(20,), dtype=float) * value
        data_a[::2] = 0.0
        data_b = np.zeros_like(data_a)
        effect_size, p = pte_stats.permutation_twosample(
            data_a=data_a,
            data_b=data_b,
            n_perm=n_perm,
            two_tailed=two_tailed,
        )
        print(f"{p=}")
        print(f"{effect_size=}")


if __name__ == "__main__":
    for two_tailed in (True, False):
        print(f"{two_tailed=}")
        # TestPermutationOnesample().test_zero_diff(two_tailed=two_tailed)
        TestPermutationTwosample().test_baseline_zero(
            value=1.0, two_tailed=two_tailed
        )
