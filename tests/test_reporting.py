import csv

from src.guv_analyzer.reporting import save_fit_results


def test_save_fit_results_creates_csv(tmp_path):
    result = {
        "baseline": 100.0,
        "amplitude": 50.0,
        "onset_s": 10.0,
        "tau_s": 8.0,
        "r_squared": 0.99,
        "residual_rmse": 1.2,
        "fit_time_s": [],
        "observed": [],
        "fitted": [],
        "residuals": [],
        "covariance": None,
    }

    output_path = tmp_path / "fit_results.csv"
    save_fit_results(result, output_path)

    assert output_path.exists()

    with output_path.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert float(rows[0]["tau_s"]) == 8.0
    assert float(rows[0]["baseline"]) == 100.0
    assert float(rows[0]["r_squared"]) == 0.99
