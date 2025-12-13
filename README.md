# Hospital Bed Management Simulation (HBMS)

A **Discrete-Event Simulation (DES)** of hospital bed management designed to analyze emergency department (ED) boarding, inpatient bed utilization, and discharge/turnover dynamics.

The model captures stochastic patient arrivals, ICU and Med-Surg bed assignment, housekeeping delays, and finite-capacity queueing behavior to study how capacity and staffing decisions affect congestion and boarding time.

---

## Project Status

**Status:** Complete

- Full simulation implemented
- Batch experiments executed
- Analysis and validation completed
- Final report and presentation delivered


## Features

### Implemented Features
- **Core Simulation Engine** (`engine.py`)  
  Custom discrete-event simulation with event scheduling, state tracking, and data export.
- **Patient & BedPool Models**  
  Finite capacity ICU and Med-Surg units with queueing and occupancy tracking.
- **HousekeepingPool**  
  Cleaning staff modeled as limited resources; governs bed turnover delays.
- **MetricsCollector**  
  Tracks admissions, discharges, failed admissions, boarding times, occupancy averages, and queue lengths.
- **Parameterized Runs**  
  Adjustable arrival rate, bed capacities, housekeeping staffing, and random seeds.
- **Automated Data Export**  
  Writes detailed event logs, time-series state snapshots, and summary metrics to `/output/` as `.csv` and `.json` files.
- **Experiment Framework**  
  - `experiments.py` -> runs a single configuration (`config.json`)  
  - `experiments_grid.py` -> runs multiple parameter combinations, aggregates results, and produces summary plots 

---

## Project Structure
HospitalBedManagementSim/
├── src/
│ ├── engine.py # core simulation logic
│ ├── experiments.py # single configuration run
│ ├── experiments_grid.py # multi-run batch experiments
│ ├── config.json # input parameters for single run
├── output/ # Generated CSV/JSON/plot results
├── requirements.txt
├── .gitignore
└── README.md

---

## Installation
**Requirements**
- Python 3.10+  
- pip

Clone the repository and install dependencies:

```bash
git clone https://github.com/tylerchetty7-ux/HospitalBedManagementSim.git
cd HospitalBedManagementSim
pip install -r requirements.txt
```

## Usage Guide

### Single Configuration Run
Runs the simulation using parameters defined in `config.json`:

```bash
python src/experiments.py --config src/config.json
```
### Batch Experiments (Parameter Grid)

Runs a grid of parameter combinations and aggregates results:

```bash
python src/experiments_grid.py
```

---

## Parameter Explanations

Simulation parameters are defined in `config.json` or internally in the experiment grid.

### Key Parameters

- **arrival_rate (λ)**  
  Mean patient arrival rate (patients per hour).  
  Higher values increase congestion and boarding time.

- **icu_beds**  
  Number of available ICU beds.  
  Strongly influences ICU boarding and queue length.

- **med_surg_beds**  
  Number of Med-Surg beds.  
  Primarily affects downstream congestion once ICU capacity is sufficient.

- **num_cleaners**  
  Number of housekeeping staff responsible for bed turnover.  
  Controls how quickly discharged beds return to service.

- **length_of_stay_distribution**  
  Stochastic distribution used to sample patient length of stay (separate for ICU and Med-Surg).

- **simulation_time**  
  Total simulated time horizon (in hours).

- **random_seed**  
  Controls stochastic reproducibility across runs.


## Example Outputs

Each simulation run produces three primary output files in the `/output/` directory:

```text
/output/
├── run_101_summary.json
├── run_101_events.csv
├── run_101_timeseries.csv
├── experiment_results.csv
├── plot_icu_occupancy.png
└── plot_icu_queue.png
```

### Sample Metrics Summary

```json
{
  "admissions": {"ICU": 48, "MED_SURG": 87},
  "discharges": {"ICU": 46, "MED_SURG": 86},
  "failed": {"ICU": 3, "MED_SURG": 2},
  "avg_boarding_time": {"ICU": 1.4, "MED_SURG": 0.8},
  "avg_occupancy": {"ICU": 0.91, "MED_SURG": 0.88}
}
```

## Notes
- `output/` is ignored by default in `.gitignore`
- Example outputs and plots were generated during batch experiments
- Results are reproducible using the provided scripts






