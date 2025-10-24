import os
import json
import itertools
from engine import SimulationEngine, MetricsCollector, BedPool, UnitType, Event, EventType
import random
import pandas as pd
import matplotlib.pyplot as plt
import argparse

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

    # Schedule initial arrival
    first_time = random.expovariate(engine.arrival_rate)
    if first_time >= engine.horizon:
        first_time = 0.0
    engine.schedule(Event(time=first_time, etype=EventType.ARRIVAL))

    engine.run()

    # Return key summary metrics for aggregation
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
    # Argument parser for config path
    parser = argparse.ArgumentParser(description="Run Hospital Bed Simulation with config file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration JSON file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = json.load(f)

    print("\n=== Running single configuration ===")
    print(json.dumps(config, indent=2))

    # Run one scenario from the configuration file
    flat = run_scenario(
        arrival_rate=config["arrival_rate"],
        icu_beds=config["icu_beds"],
        med_beds=config["med_beds"],
        cleaners=config["cleaners"],
        seed=config["seed"],
        horizon=config["horizon"]
    )

    # Save this single run to a small summary CSV
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "config_run_summary.csv")

    pd.DataFrame([flat]).to_csv(summary_path, index=False)
    print(f"\n Run complete. Results saved to {summary_path}")

