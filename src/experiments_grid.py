import os
import json
import itertools
from engine import SimulationEngine, MetricsCollector, BedPool, UnitType, Event, EventType
import random
import pandas as pd
import matplotlib.pyplot as plt

def run_scenario(arrival_rate, icu_beds, med_beds, cleaners, seed=42, horizon=100.0):
    """Run a single simulation scenario with given parameters."""
    metrics = MetricsCollector()
    icu_pool = BedPool(UnitType.ICU, capacity=icu_beds, metrics=metrics)
    med_pool = BedPool(UnitType.MED_SURG, capacity=med_beds, metrics=metrics)

    engine = SimulationEngine(
        beds_by_unit={UnitType.ICU: icu_pool, UnitType.MED_SURG: med_pool},
        metrics=metrics,
        arrival_rate=arrival_rate,
        random_seed=seed,
        horizon=horizon
    )
    engine.housekeeping.capacity = cleaners

    # schedule initial arrival
    first_time = random.expovariate(engine.arrival_rate)
    if first_time >= engine.horizon:
        first_time = 0.0
    engine.schedule(Event(time=first_time, etype=EventType.ARRIVAL))

    engine.run()

    # return key summary metrics for aggregation
    summary_path = os.path.join(engine.output_dir, f"{engine.run_id}_summary.json")
    with open(summary_path) as f:
        data = json.load(f)
    flat = {
        "run_id": data["run_id"],
        "arrival_rate": data["params"]["arrival_rate"],
        "icu_beds": data["params"]["icu_capacity"],
        "med_beds": data["params"]["med_capacity"],
        "cleaners": data["params"]["housekeepers"],
        "avg_occupancy_icu": data["metrics"]["avg_occupancy"]["ICU"],
        "avg_occupancy_med": data["metrics"]["avg_occupancy"]["MED_SURG"],
        "avg_queue_icu": data["metrics"]["avg_queue_len"]["ICU"],
        "avg_queue_med": data["metrics"]["avg_queue_len"]["MED_SURG"],
        "avg_board_icu": data["metrics"]["avg_boarding_time"]["ICU"],
        "avg_board_med": data["metrics"]["avg_boarding_time"]["MED_SURG"],
        "failed_icu": data["metrics"]["failed"]["ICU"],
        "failed_med": data["metrics"]["failed"]["MED_SURG"]
    }
    return flat


if __name__ == "__main__":
    # experiment design grid
    arrival_rates = [0.3, 0.5, 0.7]
    icu_capacities = [1, 2]
    med_capacities = [2, 3]
    cleaners = [1, 2]
    seeds = [101, 102, 103]

    scenarios = list(itertools.product(arrival_rates, icu_capacities, med_capacities, cleaners, seeds))

    results = []
    for (arr, icu, med, c, seed) in scenarios:
        print(f"\n--- Running λ={arr}, ICU={icu}, MED={med}, cleaners={c}, seed={seed} ---")
        flat = run_scenario(arrival_rate=arr, icu_beds=icu, med_beds=med, cleaners=c, seed=seed)
        results.append(flat)

    # convert results to DataFrame and save master CSV
    df = pd.DataFrame(results)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, "experiment_results.csv")
    df.to_csv(master_path, index=False)

    print(f"\n All experiments complete. Summary table saved to:\n{master_path}")

    # Data Visualization Section 

    # load results back in
    df = pd.read_csv(master_path)

    # ICU occupancy vs arrival rate
    plt.figure()
    for cleaners in sorted(df["cleaners"].unique()):
        subset = df[df["cleaners"] == cleaners]
        plt.plot(subset["arrival_rate"], subset["avg_occupancy_icu"],
                 marker="o", label=f"Cleaners={cleaners}")
    plt.title("Average ICU Occupancy vs Arrival Rate")
    plt.xlabel("Arrival Rate (λ)")
    plt.ylabel("Average ICU Occupancy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_icu_occupancy.png"))

    # ICU queue length vs arrival rate
    plt.figure()
    for cleaners in sorted(df["cleaners"].unique()):
        subset = df[df["cleaners"] == cleaners]
        plt.plot(subset["arrival_rate"], subset["avg_queue_icu"],
                 marker="o", label=f"Cleaners={cleaners}")
    plt.title("Average ICU Queue Length vs Arrival Rate")
    plt.xlabel("Arrival Rate (λ)")
    plt.ylabel("Average ICU Queue Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_icu_queue.png"))

    print("\ Plots saved to:", output_dir)
