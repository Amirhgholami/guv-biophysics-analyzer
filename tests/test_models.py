
import numpy as np
import pytest

from src.guv_analyzer.models import exponential_rise


def test_signal_before_onset_equals_baseline():
    time_s = np.array([0.0, 1.0, 2.0])

    result = exponential_rise(
        time_s,
        baseline=100.0,
        amplitude=50.0,
        onset_s=3.0,
        tau_s=2.0,
    )

    np.testing.assert_allclose(result, [100.0, 100.0, 100.0])


def test_signal_at_one_tau_reaches_63_percent():
    tau_s = 4.0
    baseline = 100.0
    amplitude = 50.0

    result = exponential_rise(
        np.array([tau_s]),
        baseline,
        amplitude,
        0.0,
        tau_s,
    )

    expected = baseline + amplitude * (1.0 - np.exp(-1.0))

    assert result[0] == pytest.approx(expected)


def test_nonpositive_tau_raises_error():
    with pytest.raises(ValueError):
        exponential_rise(
            np.array([0.0, 1.0]),
            baseline=100.0,
            amplitude=50.0,
            onset_s=0.0,
            tau_s=0.0,
        )
