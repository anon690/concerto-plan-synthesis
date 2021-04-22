import sys

from concerto_scheduling_examples.component_types import stratified_assembly
from concerto.scheduling import ComponentState, ReconfigurationProblem, SchedulingStatistics

if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        assert n > 1
    else:
        n = 3

    a = stratified_assembly(n)
    initially_active_places = {c: {c.type().get_place("running")} for c in a.instances()}
    goal_behaviors = {c: set() for c in a.instances()}
    goal_state = {}
    for c in a.instances():
        state = ComponentState(c.type())
        for p in c.type().ports():
            state.mark_inactive(p)
        goal_state[c] = state

    p = ReconfigurationProblem(a, initially_active_places, goal_behaviors, goal_state, ReconfigurationProblem.sortkey)
    solution = p.solve(verbose=True)
    SchedulingStatistics.output()
    print("solution\n")
    print(solution.script_string(explicit_waitall=False))
