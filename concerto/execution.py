from typing import Dict, Optional

from concerto.assembly import Behavior, Place, Transition


class PlaceActivationCounter:
    """This class provides a way to keep track of activation of places during a given behavior"""

    def __init__(self, b: Behavior):
        self._bhv = b
        self._counter: Dict[Place, Dict[int, int]] = {}
        for t in b.transitions():
            place, dock = t.destination()
            if place not in self._counter:
                self._counter[place] = {dock: 1}
            else:
                if dock not in self._counter[place]:
                    self._counter[place][dock] = 1
                else:
                    self._counter[place][dock] += 1

    def fire_transition(self, t: Transition) -> Optional[Place]:
        assert t in self._bhv.transitions()
        place, dock = t.destination()
        assert place in self._counter
        assert dock in self._counter[place]
        self._counter[place][dock] -= 1
        if self._counter[place][dock] < 1:
            return place
        return None
