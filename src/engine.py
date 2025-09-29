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
# quick test block to test simulation engine
# 
if __name__ == "__main__":
    sim = SimulationEngine()

    # Schedule some events
    sim.schedule(Event(time=5, etype=EventType.ARRIVAL, payload={"patient": 1}))
    sim.schedule(Event(time=2, etype=EventType.ADMIT, payload={"patient": 2}))
    sim.schedule(Event(time=8, etype=EventType.DISCHARGE, payload={"patient": 3}))

    sim.run(stop_time=10)
