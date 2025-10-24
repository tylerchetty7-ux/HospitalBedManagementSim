import heapq
from enum import Enum
import random
import numpy as np



# each event represents a discrete change in system state (arrival, admission, discharge, etc.).
class EventType(Enum):
    ARRIVAL = 1
    ADMIT = 2
    DISCHARGE = 3
    TURNOVER_DONE = 4


# event object stored in the Future Event List (FEL)
class Event:
    def __init__(self, time, etype: EventType, payload=None):
        self.time = time
        self.type = etype
        self.payload = payload or {}

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return f"Event(time={self.time}, type={self.type.name}, payload={self.payload})"


# -------------------------
# FUTURE EVENT LIST (FEL)
# -------------------------
class FutureEventList:
    def __init__(self):
        self._events = []

    def schedule(self, event: Event):
        """Add event to heap (sorted automatically by time)."""
        heapq.heappush(self._events, event)

    def next_event(self):
        """Pop the earliest scheduled event."""
        if self._events:
            return heapq.heappop(self._events)
        return None

    def peek(self):
        """Preview the next event without removing it."""
        return self._events[0] if self._events else None

    def is_empty(self):
        return len(self._events) == 0

    def __len__(self):
        return len(self._events)

    def __repr__(self):
        return f"FEL({self._events})"


# -------------------------
# HOSPITAL UNITS & PATIENTS
# -------------------------
class UnitType(Enum):
    ICU = 1
    MED_SURG = 2


# patient entity with core timestamps and LOS data.
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


# -------------------------
# METRICS COLLECTION
# -------------------------
# tracks high-level system outcomes: admissions, discharges, failures, and boarding delays.
class MetricsCollector:
    def __init__(self):
        self.admissions = {UnitType.ICU: 0, UnitType.MED_SURG: 0}
        self.discharges = {UnitType.ICU: 0, UnitType.MED_SURG: 0}
        self.failed = {UnitType.ICU: 0, UnitType.MED_SURG: 0}
        # boarding times (patients who had to wait for beds)
        self.boarding_times = {UnitType.ICU: [], UnitType.MED_SURG: []}

    def record_admission(self, patient: Patient):
        self.admissions[patient.unit] += 1

    def record_discharge(self, patient: Patient):
        self.discharges[patient.unit] += 1

    def record_failed(self, patient: Patient):
        self.failed[patient.unit] += 1

    def record_boarding(self, patient: Patient, wait_time: float):
        """Track patient wait times for queue/boarding metrics."""
        self.boarding_times[patient.unit].append(wait_time)

    def summary(self):
        """Return summary metrics for reporting."""
        base = {
            "admissions": self.admissions,
            "discharges": self.discharges,
            "failed": self.failed,
            "avg_boarding_time": {
                unit: (float(np.mean(times)) if times else 0.0)
                for unit, times in self.boarding_times.items()
            },
            "max_boarding_time": {
                unit: (float(np.max(times)) if times else 0.0)
                for unit, times in self.boarding_times.items()
            },
            "boarded_count": {
                unit: len(times) for unit, times in self.boarding_times.items()
            },
        }
        return base


# -------------------------
# BED MANAGEMENT ENTITY
# -------------------------
# handles capacity management, admissions, and occupancy counts for each unit.
class BedPool:
    def __init__(self, unit: UnitType, capacity: int, metrics: MetricsCollector = None):
        self.unit = unit
        self.capacity = capacity
        self.occupied = 0
        self.metrics = metrics

    def request_bed(self, patient: Patient, now: float) -> bool:
        """Try to admit patient if bed is available."""
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
            if self.metrics:
                self.metrics.record_failed(patient)
            print(
                f"[t={now}] ADMIT FAIL: No {self.unit.name} bed for {patient} "
                f"(Occupied={self.occupied}/{self.capacity})"
            )
            return False



# SIMULATION ENGINE
class SimulationEngine:
    def __init__(
        self,
        beds_by_unit: dict[UnitType, BedPool],
        metrics: MetricsCollector,
        arrival_rate: float = 0.5,
        random_seed: int = 42,
    ):
        # simulation clock and event list
        self.clock = 0.0
        self.fel = FutureEventList()
        self.beds = beds_by_unit
        self.metrics = metrics

        # random process parameters
        self.arrival_rate = arrival_rate  # λ for Poisson arrivals
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # distribution parameters for length of stay (LOS) and turnover
        self.los_params = {
            UnitType.ICU: {"mean": 1.6, "sigma": 0.4},
            UnitType.MED_SURG: {"mean": 1.2, "sigma": 0.3},
        }
        self.turnover_params = {
            UnitType.ICU: {"mean": 0.8, "sigma": 0.2},
            UnitType.MED_SURG: {"mean": 0.6, "sigma": 0.15},
        }

        # boarding queues (First Come, First Served)
        self.queues = {UnitType.ICU: [], UnitType.MED_SURG: []}

    def schedule(self, event: Event):
        self.fel.schedule(event)

    def sample_los(self, unit: UnitType) -> float:
        """Sample lognormal length of stay."""
        params = self.los_params[unit]
        return float(np.random.lognormal(mean=params["mean"], sigma=params["sigma"]))

    def sample_turnover(self, unit: UnitType) -> float:
        """Sample lognormal housekeeping time."""
        params = self.turnover_params[unit]
        return float(np.random.lognormal(mean=params["mean"], sigma=params["sigma"]))

    # main simulation loop
    def run(self, stop_time: float):
        """Main simulation event loop."""
        while not self.fel.is_empty():
            evt = self.fel.next_event()
            if evt.time > stop_time:
                break
            self.clock = evt.time
            print(f"\n[t={self.clock}] Handling {evt.type.name}")
            self.dispatch(evt)

    # event dispatch
    def dispatch(self, evt: Event):
        """Send event to appropriate handler."""
        if evt.type == EventType.ARRIVAL:
            self.handle_arrival(evt)
        elif evt.type == EventType.ADMIT:
            self.handle_admit(evt)
        elif evt.type == EventType.DISCHARGE:
            self.handle_discharge(evt)
        elif evt.type == EventType.TURNOVER_DONE:
            self.handle_turnover_done(evt)
        else:
            raise ValueError(f"Unknown event type: {evt.type}")

    # event handlers
    def handle_arrival(self, evt: Event):
        """Create a new patient and schedule their admission attempt."""
        unit = UnitType.ICU if random.random() < 0.3 else UnitType.MED_SURG
        pid = int(random.random() * 1e6)
        los = self.sample_los(unit)
        patient = Patient(pid=pid, unit=unit, arrival_time=self.clock, los=los)

        # immediate admission attempt
        self.schedule(Event(time=self.clock, etype=EventType.ADMIT, payload={"patient": patient}))

        # schedule next random arrival
        next_time = self.clock + random.expovariate(self.arrival_rate)
        self.schedule(Event(time=next_time, etype=EventType.ARRIVAL))

    def handle_admit(self, evt: Event):
        """Try to admit patient or enqueue if full (FCFS)."""
        patient: Patient = evt.payload["patient"]
        pool = self.beds[patient.unit]

        # FCFS: enqueue if anyone is already waiting or unit full
        if self.queues[patient.unit] or pool.occupied >= pool.capacity:
            self.queues[patient.unit].append(patient)
            print(f"[t={self.clock}] QUEUE: {patient} waiting for {patient.unit.name} bed "
                  f"(QueueLen={len(self.queues[patient.unit])})")
            return

        # admit directly (bed available)
        pool.request_bed(patient, now=self.clock)
        discharge_time = self.clock + patient.los
        self.schedule(Event(time=discharge_time, etype=EventType.DISCHARGE, payload={"patient": patient}))

    def handle_discharge(self, evt: Event):
        """Start turnover process after patient leaves."""
        patient: Patient = evt.payload["patient"]
        pool = self.beds[patient.unit]
        turnover_time = self.sample_turnover(patient.unit)
        turnover_done_time = self.clock + turnover_time

        print(f"[t={self.clock}] DISCHARGE: {patient} (Bed entering turnover for {turnover_time:.2f} hrs)")
        self.schedule(Event(time=turnover_done_time, etype=EventType.TURNOVER_DONE, payload={"unit": patient.unit}))
        pool.metrics.record_discharge(patient)

    def handle_turnover_done(self, evt: Event):
        """Free bed and admit next patient in queue if any."""
        unit = evt.payload["unit"]
        pool = self.beds[unit]

        # free capacity after cleaning
        if pool.occupied > 0:
            pool.occupied -= 1
        print(f"[t={self.clock}] TURNOVER_DONE: {unit.name} bed cleaned "
              f"(Occupied={pool.occupied}/{pool.capacity})")

        # admit next patient from queue
        if self.queues[unit]:
            next_patient = self.queues[unit].pop(0)
            wait_time = self.clock - next_patient.arrival_time
            self.metrics.record_boarding(next_patient, wait_time)
            print(f"[t={self.clock}] ADMIT FROM QUEUE: {next_patient} waited {wait_time:.2f} hrs")
            pool.request_bed(next_patient, now=self.clock)
            discharge_time = self.clock + next_patient.los
            self.schedule(Event(time=discharge_time, etype=EventType.DISCHARGE, payload={"patient": next_patient}))


# -------------------------
# SIMULATION RUN
# -------------------------
if __name__ == "__main__":
    metrics = MetricsCollector()
    icu_beds = BedPool(UnitType.ICU, capacity=1, metrics=metrics)
    med_beds = BedPool(UnitType.MED_SURG, capacity=2, metrics=metrics)

    engine = SimulationEngine(
        beds_by_unit={UnitType.ICU: icu_beds, UnitType.MED_SURG: med_beds},
        metrics=metrics,
    )

    # schedule initial arrival event to start system
    first_time = random.expovariate(engine.arrival_rate)
    engine.schedule(Event(time=first_time, etype=EventType.ARRIVAL))

    # run until simulation horizon
    engine.run(stop_time=100)

    # display end-of-run summary statistics
    print("\nMetrics summary:", metrics.summary())
