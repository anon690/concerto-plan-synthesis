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
    initially_active_places = {c: {c.type().get_place("uninstalled")} for c in a.instances()}
    goal_behaviors = {c: set() for c in a.instances()}
    u = a.get_instance("u")
    state_u = ComponentState(u.type())
    state_u.mark_active_name("service")
    goal_state = {u: state_u}

    p = ReconfigurationProblem(a, initially_active_places, goal_behaviors, goal_state, ReconfigurationProblem.sortkey)
    solution = p.solve(verbose=True)
    SchedulingStatistics.output()
    print("solution\n")
    print(solution.script_string(explicit_waitall=False))
