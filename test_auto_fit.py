from src.guv_analyzer.plotting import save_fit_plot
from src.guv_analyzer.io import load_fluorescence_csv
from src.guv_analyzer.auto_fit import find_best_fit

data = load_fluorescence_csv("data/data1.xlsx")

result = find_best_fit(
    data["time_s"].to_numpy(),
    data["fluorescence"].to_numpy(),
)

print("Automatic fit result")
print("--------------------")
print("Success:", result["success"])
print("Message:", result["message"])
print("Candidates tested:", result["candidates_tested"])
print("Successful fits:", result["successful_fits"])

if result["success"]:
    print("Fit start (s):", result["fit_start_s"])
    print("Fit end (s):", result["fit_end_s"])
    print("Points:", result["candidate_points"])
    print("Tau (s):", result["tau_s"])
    print("R-squared:", result["r_squared"])
    import pandas as pd
    save_fit_plot(data, result, "results/auto_fit.png")
