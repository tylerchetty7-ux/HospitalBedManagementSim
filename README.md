# Hospital Bed Management Simulation (HBMS)

A **Discrete-Event Simulation (DES)** of hospital bed management designed to analyze emergency department (ED) boarding, inpatient bed utilization, and discharge/turnover dynamics.  
This project models patient arrivals, ICU/Med-Surg bed assignment, cleaning delays, and admission policies to study their impact on key hospital performance metrics.

---

## 📌 Project Description
This simulation investigates how **admission control policies** (first-available, ICU reservation thresholds, cross-coverage) and **housekeeping staffing levels** affect:
- ED boarding times (mean and 90th/95th percentile)
- Unit occupancy and utilization
- Diversion hours when beds are saturated
- Queue length distributions and patient throughput

The goal is to identify configurations that balance **high bed utilization** with **low ED boarding delays**.

---

## 🛠 Features
- **Arrival Process:** Non-homogeneous Poisson process (NHPP) with time-of-day arrival rates  
- **Length of Stay:** Lognormal/Gamma LOS distributions parameterized per unit  
- **Bed Assignment:** Finite-capacity $M_t/G/s$ queue with boarding and discharge events  
- **Bed Turnover:** Stochastic cleaning delays with limited housekeeping servers  
- **Admission Policies:** Configurable (first-available, ICU reservation, cross-coverage)  
- **Metrics Collection:** Boarding time percentiles, occupancy, queue lengths, diversion hours

---

## 📂 Repository Structure
