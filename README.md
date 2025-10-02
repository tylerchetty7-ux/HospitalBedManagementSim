# Hospital Bed Management Simulation (HBMS)

A **Discrete-Event Simulation (DES)** of hospital bed management designed to analyze emergency department (ED) boarding, inpatient bed utilization, and discharge/turnover dynamics.  
This project models patient arrivals, ICU/Med-Surg bed assignment, cleaning delays, and admission policies to study their impact on key hospital performance metrics.

---

### Features
#### Implemented
- **Core Simulation Engine** (Events, Future Event List, SimulationEngine)
- **Patient & BedPool models** for ICU and Med-Surg units
- **MetricsCollector** that records admissions, discharges, and failed admissions
- Test harness showing basic patient flow and occupancy tracking

#### Future / Upcoming
- **Arrival Process:** Non-homogeneous Poisson arrivals  
- **Length of Stay:** Lognormal/Gamma LOS sampling  
- **Bed Turnover:** Cleaning delays with housekeeping staff limits  
- **Admission Policies:** First-available, ICU reservation, cross-coverage  
- **Metrics Expansion:** Boarding time percentiles, queue lengths, diversion hours

---

## Installation
Clone the repository and install dependencies:
```bash
git clone <your-repo-url>
cd HospitalBedManagementSim
pip install -r requirements.txt

## Current test harnesss
python src/engine.py

## Expected output
Admitted Patient(id=1, unit=ICU, arrival=0, los=5) to ICU. Occupied=1/1
No ICU bed available for Patient(id=2, unit=ICU, arrival=1, los=3).
Discharged Patient(id=1, unit=ICU, arrival=0, los=5) from ICU. Occupied=0/1
Admitted Patient(id=2, unit=ICU, arrival=1, los=3) to ICU. Occupied=1/1
Metrics summary: {'admissions': {ICU: 2, MED_SURG: 0},
                  'discharges': {ICU: 1, MED_SURG: 0},
                  'failed': {ICU: 1, MED_SURG: 0}}


