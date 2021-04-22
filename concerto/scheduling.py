from __future__ import annotations

import time
from itertools import chain

import z3

from concerto.assembly import Assembly, Behavior, ComponentInstance, ComponentType, Place, Port
from concerto.execution import PlaceActivationCounter
from concerto.fo_logic import *
from typing import Dict, Iterable, List, Optional, Set, Tuple, Callable, Any


class SchedulingStatistics:

    class __SchedulingStatisticsSingleton:

        def __init__(self):
            self.overall_start: float = time.perf_counter()
            self.smt_solving_time: float = 0.0
            self.unsat_solving_time: float = 0.0
            self.sat_solving_time: float = 0.0
            self.number_smt_calls: int = 0
            self.number_sat_results: int = 0
            self.number_unsat_results: int = 0
            self.number_behaviors: int = 0
            self.number_steps = 0

    __instance = __SchedulingStatisticsSingleton()

    @staticmethod
    def record_solution(s: SchedulingSolution):
        SchedulingStatistics.__instance.number_behaviors = s.number_of_behaviors()
        SchedulingStatistics.__instance.number_steps = s.number_of_steps()

    @staticmethod
    def check_with_stats(solver: z3.Optimize) -> z3.CheckSatResult:
        start: float = time.perf_counter()
        SchedulingStatistics.__instance.number_smt_calls += 1
        r = solver.check()
        solving_time = time.perf_counter() - start
        if r == z3.sat:
            SchedulingStatistics.__instance.number_sat_results += 1
            SchedulingStatistics.__instance.sat_solving_time += solving_time
        elif r == z3.unsat:
            SchedulingStatistics.__instance.number_unsat_results += 1
            SchedulingStatistics.__instance.unsat_solving_time += solving_time
        else:
            assert False
        return r

    @staticmethod
    def output() -> None:
        print(f"total time: {time.perf_counter() - SchedulingStatistics.__instance.overall_start:0.3f} seconds")
        print(f"SMT solving time: {SchedulingStatistics.__instance.sat_solving_time + SchedulingStatistics.__instance.unsat_solving_time:0.3f} seconds")
        print(f"number of call to SMT solver: {SchedulingStatistics.__instance.number_smt_calls}")
        print(f"number of sat problems: {SchedulingStatistics.__instance.number_sat_results}")
        print(f"number of unsat problems: {SchedulingStatistics.__instance.number_unsat_results}")
        if SchedulingStatistics.__instance.number_sat_results > 0:
            print(f"avg solving time (sat): {SchedulingStatistics.__instance.sat_solving_time / SchedulingStatistics.__instance.number_sat_results:0.3f} seconds")
        if SchedulingStatistics.__instance.number_unsat_results > 0:
            print(f"avg solving time (unsat): {SchedulingStatistics.__instance.unsat_solving_time / SchedulingStatistics.__instance.number_unsat_results:0.3f} seconds")
        print(f"scheduled behaviors: {SchedulingStatistics.__instance.number_behaviors}")
        print(f"synchronization steps before optimization: {SchedulingStatistics.__instance.number_steps}")


class ComponentState:

    def __init__(self, t: ComponentType):
        self._type: ComponentType = t
        self._port_states: Dict[Port, bool] = {}

    def is_active(self, p: Port) -> bool:
        return p in self._port_states and self._port_states[p]

    def is_inactive(self, p: Port) -> bool:
        return p in self._port_states and not self._port_states[p]

    def mark_active(self, p: Port) -> None:
        assert p in self._type.ports()
        assert p not in self._port_states
        self._port_states[p] = True

    def mark_active_name(self, port_name: str) -> None:
        p = self._type.get_port(port_name)
        self.mark_active(p)

    def mark_inactive(self, p: Port) -> None:
        assert p in self._type.ports()
        assert p not in self._port_states
        self._port_states[p] = False

    def mark_inactive_name(self, port_name: str) -> None:
        p = self._type.get_port(port_name)
        self.mark_inactive(p)

    def mark_active_ports(self, active_places: Iterable[Place]) -> None:
        assert all(p in self._type.places() for p in active_places)
        for port in self._type.ports():
            for place in port.bound_places():
                if place in active_places:
                    self.mark_active(port)
                    break
            if port not in self._port_states:
                self.mark_inactive(port)
        assert self.is_totally_defined()

    def is_totally_defined(self) -> bool:
        return all(p in self._port_states for p in self._type.ports())

    def component_type(self) -> ComponentType:
        return self._type

    def is_verified_by(self, active_places: Set[Place]):
        assert all(p.type() == self._type for p in active_places)
        for p in self._port_states:
            if self._port_states[p] and not any(pl in active_places for pl in p.bound_places()):
                return False
            elif not self._port_states[p] and any(pl in active_places for pl in p.bound_places()):
                return False
        return True

    def __str__(self) -> str:
        act = ""
        deact = ""
        undef = ""
        for p in self._type.ports():
            if p in self._port_states:
                if self._port_states[p]:
                    act += p.name() + "\n"
                else:
                    deact += p.name() + "\n"
            else:
                undef += p.name() + "\n"
        return ("active ports\n" + act if act else "") +\
               ("inactive ports\n" + deact if deact else "") +\
               ("undefined ports:\n" + undef if undef else "")


class BehaviorInstance:

    def __init__(self, behavior_name: str, c: ComponentInstance, final_active_places: Set[Place]):
        assert all(p in c.type().places() for p in final_active_places)
        self._behavior = c.type().get_behavior(behavior_name)
        self._component_instance: ComponentInstance = c
        self._activating_use_ports: Set[Port] = set()
        self._deactivating_provide_ports: Set[Port] = set()
        self._final_active_places = final_active_places
        self._final_state = ComponentState(self._component_instance.type())
        self._final_state.mark_active_ports(self._final_active_places)

    def behavior(self) -> Behavior:
        return self._behavior

    def component_instance(self) -> ComponentInstance:
        return self._component_instance

    def activating_use_ports(self) -> Iterable[Port]:
        return self._activating_use_ports

    def deactivating_provide_ports(self) -> Iterable[Port]:
        return self._deactivating_provide_ports

    def final_state(self) -> ComponentState:
        return self._final_state

    def final_active_places(self) -> Set[Place]:
        assert self._final_active_places is not None
        return self._final_active_places

    def mark_activating_use_port(self, p: Port) -> None:
        assert p.is_use_port()
        assert p in self._component_instance.type().use_ports()
        self._activating_use_ports.add(p)

    def mark_activating_use_ports(self, ps: Iterable[Port]) -> None:
        for p in ps:
            self.mark_activating_use_port(p)

    def mark_activating_use_port_name(self, port_name: str) -> None:
        p = self._component_instance.type().get_use_port(port_name)
        self.mark_activating_use_port(p)

    def mark_deactivating_provide_ports(self, ps: Iterable[Port]) -> None:
        for p in ps:
            self.mark_deactivating_provide_port(p)

    def mark_deactivating_provide_port(self, p: Port) -> None:
        assert p.is_provide_port()
        assert p in self._component_instance.type().provide_ports()
        self._deactivating_provide_ports.add(p)

    def mark_deactivating_provide_port_name(self, port_name: str) -> None:
        p = self._component_instance.type().get_provide_port(port_name)
        self.mark_deactivating_provide_port(p)

    def number_port_effects(self):
        return sum(1 for _ in self.deactivating_provide_ports()) + sum(1 for _ in self.activating_use_ports())

    def __str__(self):
        return "<" + self._behavior.name() + "@" + self._component_instance.id() + ">"


class SchedulingSolution:

    def __init__(self, plan: List[Set[BehaviorInstance]]):
        # the reconfiguration may not use all steps
        self._steps: List[Set[BehaviorInstance]] = [p for p in plan if len(p) > 0]
        self._barriers: List[Set[ComponentInstance]] = [{b.component_instance() for b in s} for s in self._steps]

    def number_of_steps(self) -> Int:
        return len(self._steps)

    def number_of_barriers(self) -> Int:
        return sum(len(s) for s in self._barriers)

    def number_of_behaviors(self) -> Int:
        return sum(len(s) for s in self._steps)

    def step(self, i: int) -> Set[BehaviorInstance]:
        return self._steps[i]

    def final_states(self, initial_state: Optional[Dict[ComponentInstance, ComponentState]] = None) -> Dict[ComponentInstance, ComponentState]:
        if initial_state is not None:
            f = initial_state
        else:
            f = {}
        for s in self._steps:
            for b in s:
                # we just overwrite the dictionary in the order in which behaviors are executed
                f[b.component_instance()] = b.final_state()
        return f

    def sequences(self) -> Dict[ComponentInstance, List[BehaviorInstance]]:
        seqs = {}
        for s in self._steps:
            for b in s:
                if b.component_instance() not in seqs:
                    seqs[b.component_instance()] = [b]
                else:
                    seqs[b.component_instance()].append(b)
        return seqs

    def optimize_barriers(self) -> None:
        # activated use ports and deactivated provide ports, for each component, since the last synchronization on
        # that component
        port_effects: Dict[ComponentInstance, Set[Port]] = {}
        for i in range(len(self._steps) - 1):
            step = self._steps[i]
            next_step = self._steps[i + 1]
            barrier = self._barriers[i]
            next_barrier = self._barriers[i + 1]
            for b in step:
                c = b.component_instance()
                if c not in port_effects:
                    port_effects[c] = set()
                port_effects[c].update(b.activating_use_ports())
                port_effects[c].update(b.deactivating_provide_ports())
            delayed_barriers = set()  # needed to avoid removal during iteration
            for c in barrier:
                assert c in port_effects
                may_be_delayed = True
                for p in port_effects[c]:
                    for other, other_port in c.connections(p):
                        other_behavior = [b for b in next_step if b.component_instance() == other]
                        if other_behavior:
                            assert len(other_behavior) == 1
                            if (p.is_provide_port() and other_port in other_behavior[0].activating_use_ports())\
                                    or (p.is_use_port() and other_port in other_behavior[0].deactivating_provide_ports()):
                                may_be_delayed = False
                                break
                if may_be_delayed:
                    delayed_barriers.add(c)
            next_barrier.update(delayed_barriers)
            barrier.difference_update(delayed_barriers)
            for c in barrier:
                port_effects[c] = set()

    def script_string(self, explicit_waitall=True):
        res = ""
        modified_instances = set()
        for step, barrier in zip(self._steps, self._barriers):
            for b in step:
                res += "pushB(" + b.component_instance().id() + ", " + b.behavior().name() + ")\n"
                modified_instances.add(b.component_instance())
            if not explicit_waitall and modified_instances.issubset(barrier):
                res += "waitAll()\n"
            else:
                for c in barrier:
                    res += "wait(" + c.id() + ")\n"
            modified_instances.difference_update(barrier)
        return res

    def __str__(self):
        return self.script_string(explicit_waitall=True)


class BoundedSchedulingProblem:

    def __init__(self,
                 a: Assembly,
                 initial_state: Dict[ComponentInstance, ComponentState],
                 sequences: Dict[ComponentInstance, List[BehaviorInstance]],
                 n_steps: int,
                 suggested_solution: Optional[SchedulingSolution]):
        assert n_steps > 0
        # initial state must be fully defined
        assert all(inst in initial_state for inst in a.instances())
        assert all(st.is_totally_defined() for st in initial_state.values())
        assert all(all(bhv.component_instance() == k for bhv in sequences[k]) for k in sequences.keys())
        assert suggested_solution is None or suggested_solution.number_of_steps() <= n_steps
        self._sequences = sequences
        self._assembly = a
        self._n_steps = n_steps
        self._suggested_solution = suggested_solution
        self._constraints: List[Formula] = []
        # create datatype for steps
        self._steps_dt = DatatypeSort("step")
        for i in range(n_steps):
            self._steps_dt.add_constructor("step" + str(i))
        self._steps_dt.add_constructor("final_state")
        # create datatype for behaviors
        self._bhv_dt = DatatypeSort("behavior")
        for s in sequences.values():
            for i, b in enumerate(s):
                self._bhv_dt.add_constructor(b.component_instance().id() + "_" + b.behavior().name() + "_" + str(i))
        # functions symbols and their axioms
        # schedule
        self._schedule_fun = Function("schedule", self._steps_dt, self._bhv_dt)
        # no behavior is scheduled during final step
        final_state_term = FTerm(self._steps_dt.get_constructor("final_state"))
        for bhv_fun in self._bhv_dt.constructors():
            self._constraints.append(EqualityAtomicFormula(FTerm(self._schedule_fun, FTerm(bhv_fun)),
                                                           final_state_term,
                                                           False))
        # successor
        self._succ_fun = Function("succ", self._steps_dt, self._steps_dt)
        for i in range(n_steps - 1):
            self._constraints.append(EqualityAtomicFormula(FTerm(self._succ_fun, FTerm(self._steps_dt.get_constructor("step" + str(i)))),
                                                           FTerm(self._steps_dt.get_constructor("step" + str(i + 1)))))
        self._constraints.append(EqualityAtomicFormula(FTerm(self._succ_fun, FTerm(self._steps_dt.get_constructor("step" + str(n_steps - 1)))),
                                                       FTerm(self._steps_dt.get_constructor("final_state"))))
        self._constraints.append(EqualityAtomicFormula(FTerm(self._succ_fun, FTerm(self._steps_dt.get_constructor("final_state"))),
                                                       FTerm(self._steps_dt.get_constructor("final_state"))))
        # integer value of steps
        self._toint_fun = Function("toint", IntSort(), self._steps_dt)
        for i in range(n_steps):
            self._constraints.append(EqualityAtomicFormula(FTerm(self._toint_fun, FTerm(self._steps_dt.get_constructor("step" + str(i)))),
                                                           FTerm(IntegerConstant(i))))
        self._constraints.append(EqualityAtomicFormula(FTerm(self._toint_fun, FTerm(self._steps_dt.get_constructor("final_state"))),
                                                       FTerm(IntegerConstant(n_steps))))
        # port predicates
        self._port_preds: Dict[Tuple[ComponentInstance, Port], Predicate] = {}
        for c in a.instances():
            for p in c.type().ports():
                self._port_preds[(c, p)] = Predicate("active_" + c.id() + "_" + p.name(), self._steps_dt)
        # sequentiality constraints
        for s in sequences.values():
            for i in range(len(s) - 1):
                bhv_fun1 = self._bhv_dt.get_constructor(s[i].component_instance().id() + "_" + s[i].behavior().name() + "_" + str(i))
                bhv_fun2 = self._bhv_dt.get_constructor(s[i + 1].component_instance().id() + "_" + s[i + 1].behavior().name() + "_" + str(i + 1))
                f = PredicateAtomicFormula(ArithPredicateLt(),
                                           FTerm(self._toint_fun, FTerm(self._schedule_fun, FTerm(bhv_fun1))),
                                           FTerm(self._toint_fun, FTerm(self._schedule_fun, FTerm(bhv_fun2))))
                self._constraints.append(f)
        # initial state
        for c in initial_state:
            state = initial_state[c]
            for p in c.type().ports():
                f = PredicateAtomicFormula(self._port_preds[(c, p)], FTerm(self._steps_dt.get_constructor("step0")))
                if state.is_active(p):
                    self._constraints.append(f)
                elif state.is_inactive(p):
                    self._constraints.append(NegatedFormula(f))
        # post-state constraints
        for s in sequences.values():
            for i, b in enumerate(s):
                c = b.component_instance()
                bhv_fun = self._bhv_dt.get_constructor(c.id() + "_" + b.behavior().name() + "_" + str(i))
                for p in c.type().ports():
                    f = PredicateAtomicFormula(self._port_preds[(c, p)], FTerm(self._succ_fun, FTerm(self._schedule_fun, FTerm(bhv_fun))))
                    f = f if b.final_state().is_active(p) else NegatedFormula(f)
                    self._constraints.append(f)
        # port requirement constraints
        for s in sequences.values():
            for i, b in enumerate(s):
                c = b.component_instance()
                bhv_fun = self._bhv_dt.get_constructor(c.id() + "_" + b.behavior().name() + "_" + str(i))
                for p in b.activating_use_ports():
                    if not c.is_connected(p):
                        raise Exception("use port not connected")
                    for provider, provide_port in c.connections(p):
                        providing_pred = self._port_preds[(provider, provide_port)]
                        # the provide port must be active when behavior b starts
                        f = PredicateAtomicFormula(providing_pred, FTerm(self._schedule_fun, FTerm(bhv_fun)))
                        self._constraints.append(f)
                        for j, bhv_provider in enumerate(sequences[provider]):
                            if provide_port in bhv_provider.deactivating_provide_ports():
                                # this behavior cannot run at the same time as b
                                bhv_provider_fun = self._bhv_dt.get_constructor(provider.id() + "_" + bhv_provider.behavior().name() + "_" + str(j))
                                f = EqualityAtomicFormula(FTerm(self._schedule_fun, FTerm(bhv_fun)),
                                                          FTerm(self._schedule_fun, FTerm(bhv_provider_fun)),
                                                          False)
                                self._constraints.append(f)
                for p in b.deactivating_provide_ports():
                    if not c.is_connected(p):
                        # the provide port is not connected, nothing to check
                        break
                    for user, use_port in c.connections(p):
                        using_pred = self._port_preds[(user, use_port)]
                        # the use port must be inactive when behavior b starts
                        f = NegatedFormula(PredicateAtomicFormula(using_pred, FTerm(self._schedule_fun, FTerm(bhv_fun))))
                        self._constraints.append(f)
                        for j, bhv_user in enumerate(sequences[user]):
                            if use_port in bhv_user.activating_use_ports():
                                # this behavior cannot run at the same time as b
                                bhv_provider_fun = self._bhv_dt.get_constructor(user.id() + "_" + bhv_user.behavior().name() + "_" + str(j))
                                f = EqualityAtomicFormula(FTerm(self._schedule_fun, FTerm(bhv_fun)),
                                                          FTerm(self._schedule_fun, FTerm(bhv_provider_fun)),
                                                          False)
                                self._constraints.append(f)
        # no change constraints
        for step in range(n_steps):
            step_term = FTerm(self._steps_dt.get_constructor("step" + str(step)))
            for c in a.instances():
                conj = []
                s = sequences[c]
                for i, b in enumerate(s):
                    conj.append(EqualityAtomicFormula(step_term, FTerm(self._schedule_fun, FTerm(self._bhv_dt.get_constructor(s[i].component_instance().id() + "_" + s[i].behavior().name() + "_" + str(i)))), False))
                f1 = mk_conjunction(conj)
                for p in c.type().ports():
                    f2 = EquivalenceFormula(PredicateAtomicFormula(self._port_preds[(c, p)], step_term),
                                            PredicateAtomicFormula(self._port_preds[(c, p)], FTerm(self._succ_fun, step_term)))
                    self._constraints.append(ImplicationFormula(f1, f2))

    def smtlib_string(self) -> str:
        res = "(set-logic ALL)\n"
        res += self._bhv_dt.smtlib_declaration() + "\n"
        res += self._steps_dt.smtlib_declaration() + "\n"
        res += self._schedule_fun.smtlib_declaration() + "\n"
        res += self._succ_fun.smtlib_declaration() + "\n"
        res += self._toint_fun.smtlib_declaration() + "\n"
        for p in self._port_preds.values():
            res += p.smtlib_declaration() + "\n"
        for c in self._constraints:
            res += "(assert " + c.smtlib_string() + ")\n"
        res += "(check-sat)\n"
        return res

    def _solution_from_model(self, model: z3.ModelRef) -> SchedulingSolution:
        plan: List[Set[BehaviorInstance]] = [set() for _ in range(self._n_steps)]
        for s in self._sequences.values():
            for i, b in enumerate(s):
                bhv_fun = self._bhv_dt.get_constructor(b.component_instance().id() + "_" + b.behavior().name() + "_" + str(i))
                step = model.eval(self._toint_fun.z3_func()(self._schedule_fun.z3_func()(bhv_fun.z3_func()))).as_long()
                plan[step].add(b)
        return SchedulingSolution(plan)

    def solve(self) -> Optional[SchedulingSolution]:
        s = z3.Optimize()
        # soft constraint to guide the solver towards the suggested model
        sumweight = 0
        if self._suggested_solution:
            for i in range(self._suggested_solution.number_of_steps()):
                lhs = FTerm(self._steps_dt.get_constructor("step" + str(i))).z3_expr()
                for bhv in self._suggested_solution.step(i):
                    assert bhv in self._sequences[bhv.component_instance()]
                    bhv_idx = self._sequences[bhv.component_instance()].index(bhv)
                    bhv_fun = self._bhv_dt.get_constructor(bhv.component_instance().id() + "_" + bhv.behavior().name() + "_" + str(bhv_idx))
                    rhs = FTerm(self._schedule_fun, FTerm(bhv_fun)).z3_expr()
                    s.add_soft(lhs == rhs)
                    sumweight += 1
        # soft constraint to minimize the number of steps in the schedule
        for bhv_fun in self._bhv_dt.constructors():
            lhs = FTerm(self._schedule_fun, FTerm(bhv_fun)).z3_expr()
            for step in range(1, self._n_steps):
                rhs = FTerm(self._steps_dt.get_constructor("step" + str(step))).z3_expr()
                # heavier weight for later steps
                # the weight is heavier than the sum of weight for suggested model
                s.add_soft(lhs != rhs, weight=sumweight+step)
        # hard constraints
        s.add([c.z3_expr() for c in self._constraints])

        r: z3.CheckSatResult = SchedulingStatistics.check_with_stats(s)
        return self._solution_from_model(s.model()) if r == z3.sat else None


class SchedulingProblem:

    def __init__(self,
                 a: Assembly,
                 initial_state: Dict[ComponentInstance, ComponentState],
                 sequences: Dict[ComponentInstance, List[BehaviorInstance]]):
        self._a = a
        self._initial_state = initial_state
        self._sequences = sequences

    def _trivially_unsat_extension(self, b: BehaviorInstance, prefixes: Dict[ComponentInstance, List[BehaviorInstance]]) -> bool:

        def always_active(c: ComponentInstance, p: Port) -> bool:
            return self._initial_state[c].is_active(p) and all(pfx.final_state().is_active(p) for pfx in prefixes[c])

        def always_inactive(c: ComponentInstance, p: Port) -> bool:
            return self._initial_state[c].is_inactive(p) and all(pfx.final_state().is_inactive(p) for pfx in prefixes[c])

        for use_port in b.activating_use_ports():
            if all(always_inactive(provider, provider_port) for provider, provider_port in b.component_instance().connections(use_port)):
                return True
        for provide_port in b.deactivating_provide_ports():
            if any(always_active(user, user_port) for user, user_port in b.component_instance().connections(provide_port)):
                return True
        return False

    def _trivially_sat_extension(self, b: BehaviorInstance, prefixes: Dict[ComponentInstance, List[BehaviorInstance]]) -> bool:

        def final_state_active(c: ComponentInstance, p: Port) -> bool:
            if len(prefixes[c]) > 0:
                return prefixes[c][-1].final_state().is_active(p)
            return self._initial_state[c].is_active(p)

        for use_port in b.activating_use_ports():
            if not any(final_state_active(provider, provider_port) for provider, provider_port in b.component_instance().connections(use_port)):
                return False
        for provide_port in b.deactivating_provide_ports():
            if any(final_state_active(user, user_port) for user, user_port in b.component_instance().connections(provide_port)):
                return False
        return True

    def find_maximum_schedule(self) -> SchedulingSolution:
        solution = SchedulingSolution([])
        prefixes: Dict[ComponentInstance, List[BehaviorInstance]] = {k: [] for k in self._sequences.keys()}
        suffixes: Dict[ComponentInstance, List[BehaviorInstance]] = {k: self._sequences[k].copy() for k in self._sequences.keys()}
        progress = True
        k = 0  # number of behaviors in prefixes that are not in the last computed solution
        while progress:
            progress = False
            # candidates to extend the schedule
            next_bhvs = [suffixes[c][0] for c in self._sequences.keys() if suffixes[c]]
            # prioritize those with fewer external effects
            next_bhvs.sort(key=lambda x: sum(1 for _ in chain(x.deactivating_provide_ports(), x.activating_use_ports())))
            for b in next_bhvs:
                if self._trivially_sat_extension(b, prefixes):
                    c = b.component_instance()
                    prefixes[c].append(b)
                    suffixes[c].pop(0)
                    k += 1
                    progress = True
                    if k > 5 or b == next_bhvs[-1]:
                        # * the first condition is there to avoid letting k grow too large, which can lead to a long
                        # solving time when the solver is eventually called. Instead we build the model to obtain a
                        # solution, so that solution.number_of_steps() + k is (hopefully) smaller for the next call to
                        # the solver.
                        # * the second condition ensure that the solution has been computed before returning
                        p = BoundedSchedulingProblem(self._a, self._initial_state, prefixes, solution.number_of_steps() + k, solution)
                        solution = p.solve()
                        k = 0
                        assert solution is not None
                elif not self._trivially_unsat_extension(b, prefixes):
                    c = b.component_instance()
                    prefixes[c].append(b)
                    suffixes[c].pop(0)
                    k += 1
                    n_steps = solution.number_of_steps()
                    p = BoundedSchedulingProblem(self._a, self._initial_state, prefixes, n_steps + k, solution)
                    new_solution = p.solve()
                    if new_solution is None:
                        # undo change to prefixes and suffixes
                        prefixes[c].pop()
                        suffixes[c].insert(0, b)
                        k -= 1
                    else:
                        progress = True
                        solution = new_solution
                        k = 0
        SchedulingStatistics.record_solution(solution)
        return solution


class ReconfigurationProblem:

    def __init__(self,
                 a: Assembly,
                 initially_active_places: Dict[ComponentInstance, Set[Place]],
                 goal_behaviors: Dict[ComponentInstance, Set[Behavior]],
                 goal_state: Dict[ComponentInstance, ComponentState],
                 seq_sortkey: Callable[[List[BehaviorInstance]], Any]):
        assert all(c in a.instances() for c in goal_behaviors.keys())
        assert all(c in a.instances() for c in goal_state.keys())
        assert all(c in initially_active_places for c in a.instances())

        self._assembly = a
        self._sequences = {}
        self._initial_states = {}
        self._goal_behaviors = {c: set() for c in a.instances()}
        self._goal_behaviors.update(goal_behaviors)
        self._goal_state = goal_state
        self._initially_active_places = initially_active_places
        self._complete_goal_state()
        self._seq_sortkey = seq_sortkey
        for c in a.instances():
            self._initial_states[c] = ComponentState(c.type())
            self._initial_states[c].mark_active_ports(initially_active_places[c])
            if c in goal_behaviors:
                bhvs = goal_behaviors[c]
            else:
                bhvs = set()
            if c in goal_state:
                state = goal_state[c]
            else:
                # components should remain in that state as their initial state (at least from the perspective of ports)
                state = self._initial_states[c]
            self._sequences[c] = self._behavior_sequence_from_goal(c, initially_active_places[c], bhvs, state)

    def _complete_goal_state(self):
        # define goal state for the undefined components: the reconfiguration must preserve provide port that are
        # already on
        for c in self._assembly.instances():
            if c not in self._goal_state:
                self._goal_state[c] = ComponentState(c.type())
                for p in c.type().provide_ports():
                    if p.is_active(self._initially_active_places[c]):
                        self._goal_state[c].mark_active(p)

    @staticmethod
    def sortkey(l: List[BehaviorInstance]) -> Tuple[int, int]:
        return sum(b.number_port_effects() for b in l), len(l)

    @staticmethod
    def behavior_instance_from_initial_places(b: Behavior,
                                              c: ComponentInstance,
                                              initially_active_places: Iterable[Place]) -> BehaviorInstance:
        assert all(pl in c.type().places() for pl in initially_active_places)
        active_places: Set[Place] = set(initially_active_places)
        final_places: Set[Place] = {p for p in active_places if b.is_final_place(p)}
        activating_ports: Set[Port] = set()
        deactivating_ports: Set[Port] = set()
        activation_counter = PlaceActivationCounter(b)
        while active_places:
            pl = active_places.pop()
            if b.is_final_place(pl):
                final_places.add(pl)
            else:
                for tr in b.transitions_from(pl):
                    activating_ports.update(tr.activating_ports())
                    deactivating_ports.update(tr.deactivating_ports())
                    new_active_place = activation_counter.fire_transition(tr)
                    if new_active_place is not None:
                        active_places.add(new_active_place)
        inst = BehaviorInstance(b.name(), c, final_places)
        inst.mark_activating_use_ports(p for p in activating_ports if p.is_use_port())
        inst.mark_deactivating_provide_ports(p for p in deactivating_ports if p.is_provide_port())
        return inst

    def _behavior_sequence_from_goal(self,
                                     c: ComponentInstance,
                                     initially_active_places: Set[Place],
                                     required_behaviors: Set[Behavior],
                                     target_state: ComponentState) -> List[BehaviorInstance]:
        assert all(pl in c.type().places() for pl in initially_active_places)
        assert all(bhv in c.type().behaviors() for bhv in required_behaviors)
        assert target_state.component_type() == c.type()
        # special case
        if not required_behaviors and target_state.is_verified_by(initially_active_places):
            return []
        res: List[List[BehaviorInstance]] = []
        sequence: List[BehaviorInstance] = []
        bhvs: Set[Behavior] = set(c.type().behaviors())

        def rec_enum():
            if required_behaviors.issubset(inst.behavior() for inst in sequence)\
                    and sequence and target_state.is_verified_by(sequence[-1].final_active_places()):
                res.append(sequence[:])
                return
            active_places = sequence[-1].final_active_places() if sequence else initially_active_places
            for b in c.type().behaviors():
                if b in bhvs and b.has_transitions_from(active_places):
                    inst = ReconfigurationProblem.behavior_instance_from_initial_places(b, c, active_places)
                    sequence.append(inst)
                    bhvs.remove(b)
                    rec_enum()
                    bhvs.add(b)
                    sequence.pop()

        rec_enum()
        if not res:
            raise Exception(f"reconfiguration problem for component {c.id()} has no solution")
        # pick the best solution
        return min(res, key=self._seq_sortkey)

    def _blocking_ports(self, scheduled_sequences: Dict[ComponentInstance, List[BehaviorInstance]]) -> Dict[ComponentInstance, Set[Port]]:
        blocking_ports: Dict[ComponentInstance, Set[Port]] = {k: set() for k in self._assembly.instances()}
        for c in self._sequences:
            if self._sequences[c]:
                unscheduled_bhv = self._sequences[c][0]
                if c in scheduled_sequences[c]:
                    if len(scheduled_sequences[c]) == len(self._sequences[c]):
                        break
                    else:
                        unscheduled_bhv = self._sequences[c][len(scheduled_sequences[c])]
                for p in unscheduled_bhv.activating_use_ports():
                    if not c.is_connected(p):
                        raise Exception("unconnected use port")
                    else:
                        for provider, provide_port in c.connections(p):
                            blocking_ports[provider].add(provide_port)
                for p in chain(unscheduled_bhv.deactivating_provide_ports(), unscheduled_bhv.activating_use_ports()):
                    if c.is_connected(p):
                        for other, other_port in c.connections(p):
                            blocking_ports[other].add(other_port)
                    elif p.is_use_port():
                        raise Exception("no solution, unconnected use port")
        return blocking_ports

    def solve(self, verbose=False) -> SchedulingSolution:
        progress = True
        while progress:
            progress = False
            if verbose:
                print(f"computing maximum schedule")
            s = SchedulingProblem(self._assembly, self._initial_states, self._sequences)
            solution = s.find_maximum_schedule()
            scheduled_sequences: Dict[ComponentInstance, List[BehaviorInstance]] = {c: [] for c in self._assembly.instances()}
            scheduled_sequences.update(solution.sequences())
            if verbose:
                print(f"{sum(len(l) for l in scheduled_sequences.values())}/{sum(len(l) for l in self._sequences.values())} behaviors scheduled")
            if all(len(scheduled_sequences[c]) == len(self._sequences[c]) for c in self._assembly.instances()):
                if verbose:
                    print(f"optimzing schedule")
                solution.optimize_barriers()
                if verbose:
                    print(f"synthesis completed")
                return solution
            blocking_ports = self._blocking_ports(scheduled_sequences)
            for c in blocking_ports:
                last_active_places = scheduled_sequences[c][-1].final_active_places()\
                    if scheduled_sequences[c] else self._initially_active_places[c]
                target_state = ComponentState(c.type())
                for p in blocking_ports[c]:
                    if p.is_use_port():
                        target_state.mark_inactive(p)
                    else:
                        target_state.mark_active(p)
                if not target_state.is_verified_by(last_active_places):
                    # we need to perform additional behaviors on that component
                    if len(scheduled_sequences[c]) == len(self._sequences[c]):
                        # we only add new behaviors to components for which all the behaviors were scheduled (otherwise we need to manage backtracking)
                        progress = True
                        # behaviors to get the component to the required state
                        new_bhvs = self._behavior_sequence_from_goal(c, last_active_places, set(), target_state)
                        assert new_bhvs
                        self._sequences[c].extend(new_bhvs)
                        # behaviors to complete the goal afterward (e.g. go back to the desired state)
                        last_active_places = self._sequences[c][-1].final_active_places()
                        remaining_goals_bhvs = self._goal_behaviors[c].copy()
                        remaining_goals_bhvs.difference_update(inst.behavior() for inst in self._sequences[c])
                        new_bhvs = self._behavior_sequence_from_goal(c, last_active_places, set(), self._goal_state[c])
                        self._sequences[c].extend(new_bhvs)
        raise Exception("assembly reconfiguration problem has no solution")
