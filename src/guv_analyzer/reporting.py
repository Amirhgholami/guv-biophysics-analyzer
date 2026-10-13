import csv
from pathlib import Path


def save_fit_results(result, output_path):
    """Save fitted parameters and fit-quality metrics to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "baseline",
        "amplitude",
        "onset_s",
        "tau_s",
        "r_squared",
        "residual_rmse",
        "residual_std",
        "fit_duration_s",
        "points_used",
        "tau_to_duration",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            field: result[field]
            for field in fields
        })
