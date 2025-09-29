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


# 
# quick test block to test future events list
# 
if __name__ == "__main__":
    fel = FutureEventList()

    # create some fake events
    e1 = Event(time=5, etype=EventType.ARRIVAL, payload={"patient": 1})
    e2 = Event(time=2, etype=EventType.ADMIT, payload={"patient": 2})
    e3 = Event(time=8, etype=EventType.DISCHARGE, payload={"patient": 3})

    # schedule them
    fel.schedule(e1)
    fel.schedule(e2)
    fel.schedule(e3)

    print("Initial FEL:", fel)

    # pop events in time order
    while not fel.is_empty():
        evt = fel.next_event()
        print("Processing:", evt)
