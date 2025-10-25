# Hospital Bed Management Simulation (HBMS)

A **Discrete-Event Simulation (DES)** of hospital bed management designed to analyze emergency department (ED) boarding, inpatient bed utilization, and discharge/turnover dynamics.  
This project models stochastic patient arrivals, ICU/Med-Surg bed assignment, housekeeping delays, and queueing policies to study how staffing and capacity affect key hospital performance metrics.

---

## Features

### Implemented (Milestone 3)
- **Core Simulation Engine** (`engine.py`)  
  Custom discrete-event simulation with event scheduling, state tracking, and data export.
- **Patient & BedPool Models**  
  Finite capacity ICU and Med-Surg units with queueing and occupancy tracking.
- **HousekeepingPool**  
  Cleaning staff modeled as limited resources; governs bed turnover delays.
- **MetricsCollector**  
  Tracks admissions, discharges, failed admissions, boarding times, occupancy averages, and queue lengths.
- **Parameterized Runs**  
  Adjustable arrival rate, bed capacities, and housekeeping staffing.
- **Automated Data Export**  
  Writes detailed event logs, time-series state snapshots, and summary metrics to `/output/` as `.csv` and `.json` files.
- **Experiment Framework**  
  - `experiments.py` → runs a single configuration (`config.json`)  
  - `experiments_grid.py` → runs multiple parameter combinations and produces summary plots  

### Future Enhancements (M4/M5)
- Sensitivity analysis and validation
- Policy variants (ICU reservation, cross-coverage)
- Time-varying (NHPP) arrival rates
- GUI dashboard for interactive configuration and visualization

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
Clone the repository and install dependencies:

```bash
git clone https://github.com/tylerchetty7-ux/HospitalBedManagementSim.git
cd HospitalBedManagementSim
pip install -r requirements.txt
```

## Usage
Single configuration:
```bash
python src/experiments.py --config src/config.json
```
All experiment combinations:
```bash
python src/experiments_grid.py
```
### Example output files
/output/
├── run_101_summary.json
├── run_101_events.csv
├── run_101_timeseries.csv
├── experiment_results.csv
├── plot_icu_occupancy.png
└── plot_icu_queue.png

### Sample Metrics Summary
{
  "admissions": {"ICU": 48, "MED_SURG": 87},
  "discharges": {"ICU": 46, "MED_SURG": 86},
  "failed": {"ICU": 3, "MED_SURG": 2},
  "avg_boarding_time": {"ICU": 1.4, "MED_SURG": 0.8},
  "avg_occupancy": {"ICU": 0.91, "MED_SURG": 0.88}
}

## Notes
output/ is ignored by default in .gitignore, but included for Milestone 3 submission.



