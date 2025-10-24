import heapq
from enum import Enum
import random
import numpy as np


# types of events in our simulation
class EventType(Enum):
    ARRIVAL = 1
    ADMIT = 2
    DISCHARGE = 3
    TURNOVER_DONE = 4


# event object for the Future Event List (FEL)
class Event:
    def __init__(self, time, etype: EventType, payload=None):
        self.time = time
        self.type = etype
        self.payload = payload or {}

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return f"Event(time={self.time}, type={self.type.name}, payload={self.payload})"


class FutureEventList:
    def __init__(self):
        self._events = []

    def schedule(self, event: Event):
        heapq.heappush(self._events, event)

    def next_event(self):
        if self._events:
            return heapq.heappop(self._events)
        return None

    def peek(self):
        return self._events[0] if self._events else None

    def is_empty(self):
        return len(self._events) == 0

    def __len__(self):
        return len(self._events)

    def __repr__(self):
        return f"FEL({self._events})"


# units in the hospital
class UnitType(Enum):
    ICU = 1
    MED_SURG = 2


# patient entity
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


# data collection entity (tracks admissions, discharges, and failed admissions)
class MetricsCollector:
    def __init__(self):
        self.admissions = {UnitType.ICU: 0, UnitType.MED_SURG: 0}
        self.discharges = {UnitType.ICU: 0, UnitType.MED_SURG: 0}
        self.failed = {UnitType.ICU: 0, UnitType.MED_SURG: 0}

    def record_admission(self, patient: Patient):
        self.admissions[patient.unit] += 1

    def record_discharge(self, patient: Patient):
        self.discharges[patient.unit] += 1

    def record_failed(self, patient: Patient):
        self.failed[patient.unit] += 1

    def summary(self):
        return {
            "admissions": self.admissions,
            "discharges": self.discharges,
            "failed": self.failed,
        }


# entity to request beds (uses current sim time for admit/discharge timestamps)
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
            if self.metrics:
                self.metrics.record_failed(patient)
            print(
                f"[t={now}] ADMIT FAIL: No {self.unit.name} bed for {patient} "
                f"(Occupied={self.occupied}/{self.capacity})"
            )
            return False

    def release_bed(self, patient: Patient, now: float):
        if self.occupied > 0:
            self.occupied -= 1
            patient.discharge_time = now
            if self.metrics:
                self.metrics.record_discharge(patient)
            print(
                f"[t={now}] DISCHARGE: {patient} from {self.unit.name} "
                f"(Occupied={self.occupied}/{self.capacity})"
            )


# basic simulation engine with clock, event loop, and handlers
class SimulationEngine:
    def __init__(
        self,
        beds_by_unit: dict[UnitType, BedPool],
        metrics: MetricsCollector,
        arrival_rate: float = 0.5,
        random_seed: int = 42,
    ):
        self.clock = 0.0
        self.fel = FutureEventList()
        self.beds = beds_by_unit
        self.metrics = metrics

        self.arrival_rate = arrival_rate  # λ per time unit
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # parameters for lognormal LOS sampling (correct indentation)
        self.los_params = {
            UnitType.ICU: {"mean": 1.6, "sigma": 0.4},
            UnitType.MED_SURG: {"mean": 1.2, "sigma": 0.3},
        }

    def schedule(self, event: Event):
        self.fel.schedule(event)

    def sample_los(self, unit: UnitType) -> float:
        """Draw a random length of stay from a lognormal distribution."""
        params = self.los_params[unit]
        return float(
            np.random.lognormal(mean=params["mean"], sigma=params["sigma"])
        )

    def run(self, stop_time: float):
        while not self.fel.is_empty():
            evt = self.fel.next_event()
            if evt.time > stop_time:
                break
            self.clock = evt.time
            print(f"\n[t={self.clock}] Handling {evt.type.name}")
            self.dispatch(evt)

    # only handlers implemented (no queues/turnover yet)
    def dispatch(self, evt: Event):
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

    def handle_arrival(self, evt: Event):
        # Determine unit probabilistically (e.g., 30% ICU, 70% Med/Surg)
        unit = UnitType.ICU if random.random() < 0.3 else UnitType.MED_SURG

        # Create patient with sampled LOS
        pid = int(random.random() * 1e6)  # simple unique ID
        los = self.sample_los(unit)
        patient = Patient(pid=pid, unit=unit, arrival_time=self.clock, los=los)

        # Attempt admission immediately
        self.schedule(
            Event(time=self.clock, etype=EventType.ADMIT, payload={"patient": patient})
        )

        # Schedule next arrival using exponential interarrival time
        next_time = self.clock + random.expovariate(self.arrival_rate)
        self.schedule(Event(time=next_time, etype=EventType.ARRIVAL))

    def handle_admit(self, evt: Event):
        patient: Patient = evt.payload["patient"]
        pool = self.beds[patient.unit]
        admitted = pool.request_bed(patient, now=self.clock)
        if admitted:
            # directly schedule discharge at admit_time + LOS
            discharge_time = self.clock + patient.los
            self.schedule(
                Event(
                    time=discharge_time,
                    etype=EventType.DISCHARGE,
                    payload={"patient": patient},
                )
            )
        # if not admitted, does nothing further (queues/boarding are later implemented)

    def handle_discharge(self, evt: Event):
        patient: Patient = evt.payload["patient"]
        pool = self.beds[patient.unit]
        pool.release_bed(patient, now=self.clock)

    def handle_turnover_done(self, evt: Event):
        # placeholder for testing; included to show the full lifecycle structure
        print(f"[t={self.clock}] TURNOVER_DONE (noop in Step 1) payload={evt.payload}")


# --------------- stochastic demo ----------------
if __name__ == "__main__":
    metrics = MetricsCollector()
    icu_beds = BedPool(UnitType.ICU, capacity=1, metrics=metrics)
    med_beds = BedPool(UnitType.MED_SURG, capacity=2, metrics=metrics)

    engine = SimulationEngine(
        beds_by_unit={
            UnitType.ICU: icu_beds,
            UnitType.MED_SURG: med_beds,
        },
        metrics=metrics,
    )

    # start simulation with first random arrival
    first_time = random.expovariate(engine.arrival_rate)
    engine.schedule(Event(time=first_time, etype=EventType.ARRIVAL))

    engine.run(stop_time=100)

    print("\nMetrics summary:", metrics.summary())
