import heapq
from enum import Enum

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

    # allow sorting by time when stored in a heap
    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return f"Event(time={self.time}, type={self.type.name}, payload={self.payload})"

class FutureEventList:
    def __init__(self):
        self._events = []

    def schedule(self, event: Event):
        """Add an event to the FEL (sorted by time)."""
        heapq.heappush(self._events, event)

    def next_event(self):
        """Pop the earliest event (smallest time)."""
        if self._events:
            return heapq.heappop(self._events)
        return None

    def peek(self):
        """Look at the earliest event without removing it."""
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
        return (f"Patient(id={self.id}, unit={self.unit.name}, "
                f"arrival={self.arrival_time}, los={self.los})")
# entity to request beds
class BedPool:
    def __init__(self, unit: UnitType, capacity: int):
        self.unit = unit
        self.capacity = capacity
        self.occupied = 0

    def request_bed(self, patient: Patient) -> bool:
        """Try to assign a bed to a patient. Returns True if success, False if full."""
        if self.occupied < self.capacity:
            self.occupied += 1
            patient.admit_time = patient.arrival_time  # temporary (we’ll refine later)
            print(f"Admitted {patient} to {self.unit.name}. Occupied={self.occupied}/{self.capacity}")
            return True
        else:
            print(f"No {self.unit.name} bed available for {patient}.")
            return False

    def release_bed(self, patient: Patient):
        """Release a bed when a patient is discharged."""
        if self.occupied > 0:
            self.occupied -= 1
            patient.discharge_time = patient.arrival_time + patient.los
            print(f"Discharged {patient} from {self.unit.name}. Occupied={self.occupied}/{self.capacity}")

# basic simulation engine with clock and event loop  
class SimulationEngine:
    # clock
    def __init__(self):
        self.clock = 0.0
        self.fel = FutureEventList()

    def schedule(self, event: Event):
        """Add an event to FEL."""
        self.fel.schedule(event)

    def run(self, stop_time: float):
        """Run the simulation until stop_time or FEL empty."""
        # event loop
        while not self.fel.is_empty():
            evt = self.fel.next_event()
            if evt.time > stop_time:
                break
            self.clock = evt.time
            print(f"[t={self.clock}] Handling {evt}")
            # for now just printing events


# 
# test block to test patient bed requests
# 
if __name__ == "__main__":
    # create 2 patients
    p1 = Patient(pid=1, unit=UnitType.ICU, arrival_time=0, los=5)
    p2 = Patient(pid=2, unit=UnitType.ICU, arrival_time=1, los=3)

    # ICU with 1 bed
    icu_beds = BedPool(UnitType.ICU, capacity=1)

    # try to admit both patients
    icu_beds.request_bed(p1)  # should admit
    icu_beds.request_bed(p2)  # should fail (full)

    # release p1, then admit p2 again
    icu_beds.release_bed(p1)
    icu_beds.request_bed(p2)

