import numpy as np

from src.guv_analyzer.fitting import fit_exponential_rise
from src.guv_analyzer.models import exponential_rise


def test_fit_recovers_known_tau():
    time_s = np.linspace(0, 30, 301)

    true_baseline = 100.0
    true_amplitude = 50.0
    true_onset = 5.0
    true_tau = 4.0

    fluorescence = exponential_rise(
        time_s,
        true_baseline,
        true_amplitude,
        true_onset,
        true_tau,
    )

    result = fit_exponential_rise(time_s, fluorescence)

    assert abs(result["tau_s"] - true_tau) < 0.2
    assert abs(result["onset_s"] - true_onset) < 0.2
    assert result["r_squared"] > 0.999
