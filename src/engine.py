import heapq
from enum import Enum
import random
import numpy as np
from collections import deque
import csv, json, os


# ============================
# EVENT TYPES
# ============================
class EventType(Enum):
    ARRIVAL = 1
    ADMIT = 2
    DISCHARGE = 3
    TURNOVER_DONE = 4


# ============================
# EVENT OBJECT
# ============================
class Event:
    def __init__(self, time, etype: EventType, payload=None):
        self.time = time
        self.type = etype
        self.payload = payload or {}

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return f"Event(time={self.time}, type={self.type.name}, payload={self.payload})"


# ============================
# FUTURE EVENT LIST (FEL)
# ============================
class FutureEventList:
    def __init__(self):
        self._events = []

    def schedule(self, event: Event):
        heapq.heappush(self._events, event)

    def next_event(self):
        return heapq.heappop(self._events) if self._events else None

    def is_empty(self):
        return len(self._events) == 0


# ============================
# UNIT & PATIENT ENTITIES
# ============================
class UnitType(Enum):
    ICU = 1
    MED_SURG = 2


class Patient:
    def __init__(self, pid: int, unit: UnitType, arrival_time: float, los: float):
        self.id = pid
        self.unit = unit
        self.arrival_time = arrival_time
        self.los = los
        self.admit_time = None
        self.discharge_time = None

    def __repr__(self):
        return (
            f"Patient(id={self.id}, unit={self.unit.name}, "
            f"arrival={self.arrival_time}, los={self.los})"
        )


# ============================
# METRICS COLLECTION
# ============================
class MetricsCollector:
    def __init__(self):
        self.admissions = {UnitType.ICU: 0, UnitType.MED_SURG: 0}
        self.discharges = {UnitType.ICU: 0, UnitType.MED_SURG: 0}
        self.failed = {UnitType.ICU: 0, UnitType.MED_SURG: 0}
        self.boarding_times = {UnitType.ICU: [], UnitType.MED_SURG: []}
        self.last_time = 0.0
        self.occ_area = {UnitType.ICU: 0.0, UnitType.MED_SURG: 0.0}
        self.q_area = {UnitType.ICU: 0.0, UnitType.MED_SURG: 0.0}
        self.max_occ = {UnitType.ICU: 0, UnitType.MED_SURG: 0}
        self.max_queue = {UnitType.ICU: 0, UnitType.MED_SURG: 0}

    def record_admission(self, patient: Patient):
        self.admissions[patient.unit] += 1

    def record_discharge(self, patient: Patient):
        self.discharges[patient.unit] += 1

    def record_failed(self, patient: Patient):
        self.failed[patient.unit] += 1

    def record_boarding(self, patient: Patient, wait_time: float):
        self.boarding_times[patient.unit].append(wait_time)

    def update_state_areas(self, now: float, beds: dict, queues: dict):
        dt = now - self.last_time
        if dt < 0:
            return
        for u in beds:
            self.occ_area[u] += beds[u].occupied * dt
            self.q_area[u] += len(queues[u]) * dt
            self.max_occ[u] = max(self.max_occ[u], beds[u].occupied)
            self.max_queue[u] = max(self.max_queue[u], len(queues[u]))
        self.last_time = now

    def summary(self):
        return {
            "admissions": self.admissions,
            "discharges": self.discharges,
            "failed": self.failed,
            "avg_boarding_time": {
                u: (float(np.mean(v)) if v else 0.0) for u, v in self.boarding_times.items()
            },
            "max_boarding_time": {
                u: (float(np.max(v)) if v else 0.0) for u, v in self.boarding_times.items()
            },
            "boarded_count": {u: len(v) for u, v in self.boarding_times.items()},
            "avg_occupancy": {
                u: round(self.occ_area[u] / max(self.last_time, 1e-9), 3) for u in self.occ_area
            },
            "avg_queue_len": {
                u: round(self.q_area[u] / max(self.last_time, 1e-9), 3) for u in self.q_area
            },
            "max_occupancy": self.max_occ,
            "max_queue_len": self.max_queue,
            "sim_time": round(self.last_time, 3),
        }


# ============================
# BED & HOUSEKEEPING
# ============================
class BedPool:
    def __init__(self, unit: UnitType, capacity: int, metrics: MetricsCollector = None):
        self.unit = unit
        self.capacity = capacity
        self.occupied = 0
        self.metrics = metrics

    def request_bed(self, patient: Patient, now: float) -> bool:
        if self.occupied < self.capacity:
            self.occupied += 1
            patient.admit_time = now
            if self.metrics:
                self.metrics.record_admission(patient)
            print(
                f"[t={now}] ADMIT: {patient} -> {self.unit.name} "
                f"(Occupied={self.occupied}/{self.capacity})"
            )
            return True
        else:
            print(
                f"[t={now}] ADMIT FAIL: No {self.unit.name} bed for {patient} "
                f"(Occupied={self.occupied}/{self.capacity})"
            )
            return False


class HousekeepingPool:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.busy = 0
        self.active_cleanings = []
        self.wait_queue = deque()

    def _start_cleaning_now(self, unit: UnitType, now: float):
        self.busy += 1
        self.active_cleanings.append((unit, now))

    def request_cleaner(self, unit: UnitType, turnover_time: float, now: float) -> bool:
        if self.busy < self.capacity:
            self._start_cleaning_now(unit, now)
            return True
        else:
            self.wait_queue.append((unit, turnover_time))
            return False

    def release_cleaner(self, now: float):
        if self.busy > 0:
            self.busy -= 1
        if self.active_cleanings:
            self.active_cleanings.pop(0)
        if self.wait_queue:
            next_unit, next_turnover = self.wait_queue.popleft()
            self._start_cleaning_now(next_unit, now)
            return (next_unit, next_turnover)
        return None


# ============================
# SIMULATION ENGINE
# ============================
class SimulationEngine:
    def __init__(
        self,
        beds_by_unit: dict[UnitType, BedPool],
        metrics: MetricsCollector,
        arrival_rate: float = 0.5,
        random_seed: int = 42,
        horizon: float = 100.0,
        warmdown: float | None = None,
    ):
        self.clock = 0.0
        self.fel = FutureEventList()
        self.beds = beds_by_unit
        self.metrics = metrics

        self.arrival_rate = arrival_rate
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.los_params = {
            UnitType.ICU: {"mean": 1.6, "sigma": 0.4},
            UnitType.MED_SURG: {"mean": 1.2, "sigma": 0.3},
        }
        self.turnover_params = {
            UnitType.ICU: {"mean": 0.8, "sigma": 0.2},
            UnitType.MED_SURG: {"mean": 0.6, "sigma": 0.15},
        }

        self.queues = {UnitType.ICU: deque(), UnitType.MED_SURG: deque()}
        self.max_queue_len = {UnitType.ICU: 15, UnitType.MED_SURG: 20}
        self.max_wait_hours = {UnitType.ICU: 48.0, UnitType.MED_SURG: 24.0}
        self.housekeeping = HousekeepingPool(capacity=2)

        self.horizon = horizon
        self.warmdown_limit = warmdown
        self.arrivals_open = True
        self.last_event_time = 0.0

        self.event_log = []
        self.state_log = []
        self.log_interval = 1.0
        self.next_log_time = 0.0

        # ✅ Fixed path: save to project root
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
        os.makedirs(self.output_dir, exist_ok=True)

        self.run_id = f"run_{self.random_seed}"

    def schedule(self, event: Event):
        self.fel.schedule(event)

    def sample_los(self, unit: UnitType) -> float:
        params = self.los_params[unit]
        return float(np.random.lognormal(mean=params["mean"], sigma=params["sigma"]))

    def sample_turnover(self, unit: UnitType) -> float:
        params = self.turnover_params[unit]
        return float(np.random.lognormal(mean=params["mean"], sigma=params["sigma"]))

    def run(self):
        while not self.fel.is_empty():
            evt = self.fel.next_event()
            if self.warmdown_limit is not None and evt.time > (self.horizon + self.warmdown_limit):
                break

            self.clock = evt.time
            self.last_event_time = self.clock
            self.metrics.update_state_areas(self.clock, self.beds, self.queues)

            if self.arrivals_open and self.clock >= self.horizon:
                self.arrivals_open = False

            print(f"\n[t={self.clock}] Handling {evt.type.name}")

            self.event_log.append({
                "time": round(self.clock, 3),
                "event_type": evt.type.name,
                "unit": evt.payload.get("patient", {}).unit.name
                    if isinstance(evt.payload.get("patient"), Patient)
                    else evt.payload.get("unit", None).name
                    if evt.payload.get("unit") else None,
                "patient_id": getattr(evt.payload.get("patient"), "id", None)
            })

            self.dispatch(evt)
            self.metrics.update_state_areas(self.last_event_time, self.beds, self.queues)

            if self.clock >= self.next_log_time:
                self.state_log.append({
                    "time": round(self.clock, 3),
                    "icu_occupied": self.beds[UnitType.ICU].occupied,
                    "med_occupied": self.beds[UnitType.MED_SURG].occupied,
                    "icu_queue": len(self.queues[UnitType.ICU]),
                    "med_queue": len(self.queues[UnitType.MED_SURG]),
                    "housekeepers_busy": self.housekeeping.busy
                })
                self.next_log_time += self.log_interval

        self.export_results()

    def dispatch(self, evt: Event):
        if evt.type == EventType.ARRIVAL:
            self.handle_arrival(evt)
        elif evt.type == EventType.ADMIT:
            self.handle_admit(evt)
        elif evt.type == EventType.DISCHARGE:
            self.handle_discharge(evt)
        elif evt.type == EventType.TURNOVER_DONE:
            self.handle_turnover_done(evt)

    # ---- Event handlers ----
    def handle_arrival(self, evt: Event):
        unit = UnitType.ICU if random.random() < 0.3 else UnitType.MED_SURG
        pid = int(random.random() * 1e6)
        los = self.sample_los(unit)
        patient = Patient(pid=pid, unit=unit, arrival_time=self.clock, los=los)
        self.schedule(Event(time=self.clock, etype=EventType.ADMIT, payload={"patient": patient}))

        if self.arrivals_open:
            next_time = self.clock + random.expovariate(self.arrival_rate)
            if next_time < self.horizon:
                self.schedule(Event(time=next_time, etype=EventType.ARRIVAL))

    def handle_admit(self, evt: Event):
        patient: Patient = evt.payload["patient"]
        pool = self.beds[patient.unit]
        queue = self.queues[patient.unit]

        if queue or pool.occupied >= pool.capacity:
            if len(queue) >= self.max_queue_len[patient.unit]:
                self.metrics.record_failed(patient)
                print(f"[t={self.clock}] DIVERTED: {patient} due to full queue (len={len(queue)})")
                return
            queue.append(patient)
            print(f"[t={self.clock}] QUEUE: {patient} waiting for {patient.unit.name} bed (QueueLen={len(queue)})")
            return

        pool.request_bed(patient, now=self.clock)
        self.schedule(Event(time=self.clock + patient.los, etype=EventType.DISCHARGE, payload={"patient": patient}))

    def handle_discharge(self, evt: Event):
        patient: Patient = evt.payload["patient"]
        pool = self.beds[patient.unit]
        self.metrics.record_discharge(patient)
        patient.discharge_time = self.clock
        turnover_time = self.sample_turnover(patient.unit)
        print(f"[t={self.clock}] DISCHARGE: {patient} (Bed entering turnover for {turnover_time:.2f} hrs)")
        started = self.housekeeping.request_cleaner(patient.unit, turnover_time, now=self.clock)
        if started:
            self.schedule(Event(time=self.clock + turnover_time, etype=EventType.TURNOVER_DONE, payload={"unit": patient.unit}))
        else:
            print(f"[t={self.clock}] All cleaners busy — {patient.unit.name} bed queued for cleaning.")

    def handle_turnover_done(self, evt: Event):
        unit = evt.payload["unit"]
        pool = self.beds[unit]
        if pool.occupied > 0:
            pool.occupied -= 1
        print(f"[t={self.clock}] TURNOVER_DONE: {unit.name} bed cleaned (Occupied={pool.occupied}/{pool.capacity})")

        next_job = self.housekeeping.release_cleaner(now=self.clock)
        if next_job:
            next_unit, next_turnover = next_job
            next_done_time = self.clock + next_turnover
            print(f"[t={self.clock}] HOUSEKEEPER reassigned → new {next_unit.name} cleaning (will finish at {next_done_time:.2f})")
            self.schedule(Event(time=next_done_time, etype=EventType.TURNOVER_DONE, payload={"unit": next_unit}))

        if self.queues[unit]:
            next_patient = self.queues[unit].popleft()
            wait_time = self.clock - next_patient.arrival_time
            if wait_time > self.max_wait_hours[unit]:
                self.metrics.record_failed(next_patient)
                print(f"[t={self.clock}] DIVERTED (timeout): {next_patient} waited {wait_time:.2f} hrs")
                return
            self.metrics.record_boarding(next_patient, wait_time)
            print(f"[t={self.clock}] ADMIT FROM QUEUE: {next_patient} waited {wait_time:.2f} hrs")
            pool.request_bed(next_patient, now=self.clock)
            self.schedule(Event(time=self.clock + next_patient.los, etype=EventType.DISCHARGE, payload={"patient": next_patient}))

    def export_results(self):
        """Write event log, time-series, and summary to CSV/JSON."""
        if not self.event_log:
            print("[WARN] No events logged; nothing to export.")
            return

        # 1. Events CSV
        events_path = os.path.join(self.output_dir, f"{self.run_id}_events.csv")
        with open(events_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.event_log[0].keys())
            writer.writeheader()
            writer.writerows(self.event_log)

        # 2. Time-series CSV
        if self.state_log:
            ts_path = os.path.join(self.output_dir, f"{self.run_id}_timeseries.csv")
            with open(ts_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.state_log[0].keys())
                writer.writeheader()
                writer.writerows(self.state_log)

        # 3. Summary JSON — convert Enum keys to strings
        metrics_summary = {}
        for key, val in self.metrics.summary().items():
            if isinstance(val, dict):
                # Convert Enum keys (UnitType.ICU → "ICU")
                metrics_summary[key] = {
                    (u.name if isinstance(u, Enum) else u): v for u, v in val.items()
                }
            else:
                metrics_summary[key] = val

        summary_path = os.path.join(self.output_dir, f"{self.run_id}_summary.json")
        summary_data = {
            "run_id": self.run_id,
            "params": {
                "arrival_rate": self.arrival_rate,
                "icu_capacity": self.beds[UnitType.ICU].capacity,
                "med_capacity": self.beds[UnitType.MED_SURG].capacity,
                "housekeepers": self.housekeeping.capacity,
                "horizon": self.horizon,
                "seed": self.random_seed
            },
            "metrics": metrics_summary
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"\n[Run {self.run_id}] Data exported to {self.output_dir}")



# ============================
# MAIN EXECUTION
# ============================
if __name__ == "__main__":
    metrics = MetricsCollector()
    icu_beds = BedPool(UnitType.ICU, capacity=1, metrics=metrics)
    med_beds = BedPool(UnitType.MED_SURG, capacity=2, metrics=metrics)

    engine = SimulationEngine(
        beds_by_unit={UnitType.ICU: icu_beds, UnitType.MED_SURG: med_beds},
        metrics=metrics,
        horizon=100.0,
        warmdown=None
    )

    engine.housekeeping.capacity = 2

    first_time = random.expovariate(engine.arrival_rate)
    if first_time >= engine.horizon:
        first_time = 0.0
    engine.schedule(Event(time=first_time, etype=EventType.ARRIVAL))

    engine.run()
    print("\nMetrics summary:", metrics.summary())
