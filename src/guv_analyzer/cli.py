
import argparse
from pathlib import Path

from .fitting import fit_exponential_rise
from .io import load_fluorescence_csv
from .plotting import save_fit_plot


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GUV fluorescence time-series data."
    )

    parser.add_argument(
        "input_csv",
        help="CSV file with time_s and fluorescence columns",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start of fitting interval in seconds",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End of fitting interval in seconds",
    )

    parser.add_argument(
        "--plot",
        default="results/fluorescence_fit.png",
        help="Path for the output plot",
    )

    args = parser.parse_args()

    data = load_fluorescence_csv(args.input_csv)

    result = fit_exponential_rise(
        data["time_s"].to_numpy(),
        data["fluorescence"].to_numpy(),
        fit_start=args.start,
        fit_end=args.end,
    )

    output_path = Path(args.plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_fit_plot(data, result, output_path)

    print("\nExponential-rise fit")
    print("--------------------")
    print(f"Baseline:      {result['baseline']:.4f}")
    print(f"Amplitude:     {result['amplitude']:.4f}")
    print(f"Onset time:    {result['onset_s']:.4f} s")
    print(f"Tau:           {result['tau_s']:.4f} s")
    print(f"R-squared:     {result['r_squared']:.5f}")
    print(f"Residual RMSE: {result['residual_rmse']:.4f}")
    print(f"Plot saved to: {output_path}")

    print(
        "\nNote: tau describes the fitted fluorescence response. "
        "It is not automatically a water-permeability measurement."
    )


if __name__ == "__main__":
    main()
