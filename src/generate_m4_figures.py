import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_figures():
    input_csv = r"C:\Projects\HospitalBedManagementSim\output\experiment_results.csv"
    output_dir = r"C:\Projects\HospitalBedManagementSim\output\m4_outputs"

    # create output directory if missing
    os.makedirs(output_dir, exist_ok=True)

    # load CSV
    df = pd.read_csv(input_csv)

    # -------------------------------------------------------
    # figure 1: arrival rate → mean ICU boarding time
    # -------------------------------------------------------
    arr_rate_group = df.groupby("arrival_rate")["avg_board_icu"].mean()

    plt.figure(figsize=(6, 4))
    arr_rate_group.plot(kind="bar")
    plt.title("Effect of Arrival Rate on ICU Boarding Time")
    plt.xlabel("Arrival Rate (patients/hour)")
    plt.ylabel("Mean ICU Boarding Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_arrival_rate_vs_icu_boarding.png"))
    plt.close()

    # -------------------------------------------------------
    # figure 2: ICU capacity → mean ICU boarding time
    # -------------------------------------------------------
    icu_cap_group = df.groupby("icu_beds")["avg_board_icu"].mean()

    plt.figure(figsize=(6, 4))
    icu_cap_group.plot(kind="bar")
    plt.title("Effect of ICU Capacity on ICU Boarding Time")
    plt.xlabel("ICU Bed Count")
    plt.ylabel("Mean ICU Boarding Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_icu_capacity_vs_icu_boarding.png"))
    plt.close()

    # -------------------------------------------------------
    # figure 3: scenario comparison (A–D)
    # -------------------------------------------------------
    # scenario logic
    def assign_scenario(row):
        if row["arrival_rate"] == 0.4 and row["icu_beds"] == 3 and row["med_beds"] == 4 and row["cleaners"] == 2:
            return "A"
        if row["arrival_rate"] == 0.6 and row["icu_beds"] == 2 and row["med_beds"] == 3 and row["cleaners"] == 1:
            return "B"
        if row["arrival_rate"] == 0.6 and row["icu_beds"] == 3 and row["med_beds"] == 3 and row["cleaners"] == 2:
            return "C"
        if row["arrival_rate"] == 0.6 and row["icu_beds"] == 3 and row["med_beds"] == 4 and row["cleaners"] == 3:
            return "D"
        return None

    df["scenario"] = df.apply(assign_scenario, axis=1)

    scenario_df = df[df["scenario"].notnull()].groupby("scenario")["avg_board_icu"].mean()

    plt.figure(figsize=(6, 4))
    scenario_df.plot(kind="bar")
    plt.title("Scenario Comparison: Mean ICU Boarding Time")
    plt.xlabel("Scenario")
    plt.ylabel("Mean ICU Boarding Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_scenario_comparison.png"))
    plt.close()

    print("Figures generated at:", output_dir)


if __name__ == "__main__":
    generate_figures()
