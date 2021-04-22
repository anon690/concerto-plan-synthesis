from __future__ import annotations

import operator
from abc import abstractmethod
from typing import Dict, List, Iterable

from z3 import *


class Sort:

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def z3_sort(self):
        pass


class BoolSort(Sort):

    def __eq__(self, other):
        return isinstance(other, BoolSort)

    def name(self):
        return "Bool"

    def z3_sort(self):
        return z3.BoolSort()


class IntSort(Sort):

    def __eq__(self, other):
        return isinstance(other, IntSort)

    def name(self):
        return "Int"

    def z3_sort(self):
        return z3.IntSort()


class UninterpretedSort(Sort):

    def __init__(self, name: str):
        self._name = name
        self._z3_sort = z3.DeclareSort(name)

    def name(self):
        return self._name

    def smtlib_declaration(self):
        return "(declare-sort " + self._name + " 0)\n"

    def z3_sort(self):
        return self._z3_sort


class DatatypeSort(Sort):

    def __init__(self, name: str):
        self._name = name
        self._z3_sort = None
        self._constructors: Dict[str, DatatypeConstructor] = {}

    def name(self):
        return self._name

    def add_constructor(self, name: str, *sig: Sort):
        assert name not in self._constructors
        self._constructors[name] = DatatypeConstructor(name, self, *sig)

    def get_constructor(self, name: str) -> DatatypeConstructor:
        assert name in self._constructors
        return self._constructors[name]

    def constructors(self) -> Iterable[DatatypeConstructor]:
        return self._constructors.values()

    def smtlib_declaration(self):
        return "(declare-datatypes ((" + self._name + " 0)) ((" +\
               " ".join(map(lambda c: c.smtlib_declaration(), self._constructors.values())) + ")))"

    def z3_sort(self):
        if self._z3_sort is None:
            dt = z3.Datatype(self._name)
            for c in self._constructors.values():
                proj_funs = []
                for i in range(c.n_args()):
                    proj_funs.append(("proj_" + c.name() + "_" + str(i), c.nth_arg(i).z3_sort()))
                dt.declare(c.name(), *proj_funs)
            self._z3_sort = dt.create()
        return self._z3_sort


class Predicate:

    def __init__(self, name: str, *sig: Sort):
        self._name = name
        self._sig = sig

    def name(self) -> str:
        return self._name

    def n_args(self) -> int:
        return len(self._sig)

    def nth_arg(self, n) -> Sort:
        assert n < len(self._sig)
        return self._sig[n]

    def smtlib_declaration(self) -> str:
        return "(declare-fun " + self._name + " (" + " ".join(map(lambda x: x.name(), self._sig)) + ") Bool)"

    def z3_func(self):
        return z3.Function(self._name, list(map(lambda x: x.z3_sort(), self._sig)) + [z3.BoolSort()])


class ArithPredicateLt(Predicate):

    def __init__(self):
        super().__init__("<", IntSort(), IntSort())

    def z3_func(self):
        return operator.lt


class Function:

    def __init__(self, name: str, rng: Sort, *sig: Sort):
        self._name = name
        self._rng = rng
        self._sig = sig

    def name(self) -> str:
        return self._name

    def range(self) -> Sort:
        return self._rng

    def n_args(self) -> int:
        return len(self._sig)

    def nth_arg(self, n) -> Sort:
        assert n < len(self._sig)
        return self._sig[n]

    def smtlib_declaration(self):
        return "(declare-fun " + self._name + " (" +\
               " ".join(map(lambda x: x.name(), self._sig)) + ") " + self._rng.name() + ")"

    def z3_func(self):
        return z3.Function(self._name, list(map(lambda x: x.z3_sort(), self._sig)) + [self._rng.z3_sort()])


class DatatypeConstructor(Function):

    def smtlib_declaration(self):
        res = "(" + self._name
        for i in range(len(self._sig)):
            res += " (proj_" + self._name + "_" + str(i) + " " + self._sig[i].name() + ")"
        return res + ")"

    def z3_func(self):
        return getattr(self._rng.z3_sort(), self._name)


class IntegerConstant(Function):

    def __init__(self, i: int):
        super().__init__(str(i), IntSort())
        self._i = i

    def smtlib_declaration(self) -> str:
        return ""

    def z3_func(self):
        return self._i


class Term:

    @abstractmethod
    def sort(self) -> Sort:
        pass

    @abstractmethod
    def smtlib_string(self) -> str:
        pass

    @abstractmethod
    def z3_expr(self):
        pass


class FTerm(Term):
    def __init__(self, f: Function, *args: Term):
        assert len(args) == f.n_args()
        for i in range(f.n_args()):
            assert args[i].sort() == f.nth_arg(i)
        self._f = f
        self._args = args

    def sort(self):
        return self._f.range()

    def smtlib_string(self):
        if not self._args:
            return self._f.name()
        else:
            return "(" + self._f.name() + " " + " ".join(map(lambda a: a.smtlib_string(), self._args)) + ")"

    def z3_expr(self):
        if self._args:
            return self._f.z3_func()(*map(lambda x: x.z3_expr(), self._args))
        return self._f.z3_func()


class VTerm(Term):

    fresh_var_id = 0

    def __init__(self, s: Sort):
        self._id = VTerm.fresh_var_id
        self._sort = s
        VTerm.fresh_var_id += 1

    def __eq__(self, other):
        if isinstance(other, VTerm):
            assert self._id != other._id or self._sort == other._sort
            return self._id == other._id
        return False

    def __hash__(self):
        return hash(self._id)

    def sort(self):
        return self._sort

    def smtlib_string(self):
        return "x" + str(self._id)

    def z3_expr(self):
        # z3 variables are for open formulas, whereas we use quantifiers
        return z3.Const("x" + str(self._id), self._sort.z3_sort())


class Formula:

    @abstractmethod
    def smtlib_string(self) -> str:
        pass

    @abstractmethod
    def z3_expr(self):
        pass


class TruthFormula(Formula):

    def smtlib_string(self):
        return "true"

    def z3_expr(self) -> z3.ExprRef:
        return z3.BoolVal(True)


class EqualityAtomicFormula(Formula):

    def __init__(self, e1: Term, e2: Term, polarity=True):
        assert e1.sort() == e2.sort()
        self._e1 = e1
        self._e2 = e2
        self._polarity = polarity

    def smtlib_string(self):
        if self._polarity:
            return "(= " + self._e1.smtlib_string() + " " + self._e2.smtlib_string() + ")"
        else:
            return "(distinct " + self._e1.smtlib_string() + " " + self._e2.smtlib_string() + ")"

    def z3_expr(self):
        if self._polarity:
            return self._e1.z3_expr() == self._e2.z3_expr()
        else:
            return self._e1.z3_expr() != self._e2.z3_expr()


class PredicateAtomicFormula(Formula):

    def __init__(self, p: Predicate, *args: Term):
        assert len(args) == p.n_args()
        for i in range(len(args)):
            assert args[i].sort() == p.nth_arg(i)
        self._p = p
        self._args = args

    def smtlib_string(self):
        if not self._args:
            return self._p.name()
        else:
            return "(" + self._p.name() + " " + " ".join(map(lambda a: a.smtlib_string(), self._args)) + ")"

    def z3_expr(self):
        if self._args:
            return self._p.z3_func()(*map(lambda x: x.z3_expr(), self._args))
        return self._p.z3_func()


class NegatedFormula(Formula):

    def __init__(self, f: Formula):
        self._f = f

    def smtlib_string(self):
        return "(not " + self._f.smtlib_string() + ")"

    def z3_expr(self):
        return z3.Not(self._f.z3_expr())


class ConjunctionFormula(Formula):

    def __init__(self, fargs: List[Formula]):
        assert len(fargs) > 1
        self._fargs: List[Formula] = fargs

    def smtlib_string(self):
        return "(and " + " ".join(map(lambda f: f.smtlib_string(), self._fargs)) + ")"

    def z3_expr(self):
        return z3.And(list(map(lambda x: x.z3_expr(), self._fargs)))


class DisjunctionFormula(Formula):

    def __init__(self, fargs: List[Formula]):
        assert len(fargs) > 1
        self._fargs: List[Formula] = fargs

    def smtlib_string(self):
        return "(or " + " ".join(map(lambda f: f.smtlib_string(), self._fargs)) + ")"

    def z3_expr(self):
        return z3.Or(list(map(lambda f: f.z3_expr(), self._fargs)))


class ImplicationFormula(Formula):

    def __init__(self, f1: Formula, f2: Formula):
        self._f1: Formula = f1
        self._f2: Formula = f2

    def smtlib_string(self):
        return "(=> " + self._f1.smtlib_string() + " " + self._f2.smtlib_string() + ")"

    def z3_expr(self):
        return z3.Implies(self._f1.z3_expr(), self._f2.z3_expr())


class EquivalenceFormula(Formula):

    def __init__(self, f1: Formula, f2: Formula):
        self._f1 = f1
        self._f2 = f2

    def smtlib_string(self):
        return "(= " + self._f1.smtlib_string() + " " + self._f2.smtlib_string() + ")"

    def z3_expr(self):
        return self._f1.z3_expr().__eq__(self._f2.z3_expr())


class UniversalFormula(Formula):

    def __init__(self, f: Formula, *vs: VTerm):
        assert len(vs) > 0
        self._f = f
        self._vars = vs

    def smtlib_string(self):
        return "(forall ("\
               + " ".join(map(lambda x: "(" + x.smtlib_string() + " " + x.sort().name() + ")", self._vars))\
               + ") " + self._f.smtlib_string() + ")"

    def z3_expr(self):
        return z3.ForAll(list(map(lambda f: f.z3_expr(), self._vars)), self._f.z3_expr())


class ExistentialFormula(Formula):

    def __init__(self, f: Formula, *vs: VTerm):
        assert len(vs) > 0
        self._f = f
        self._vars = vs

    def smtlib_string(self):
        return "(exists ("\
               + " ".join(map(lambda x: "(" + x.smtlib_string() + " " + x.sort().name() + ")", self._vars))\
               + ") " + self._f.smtlib_string() + ")"

    def z3_expr(self):
        return z3.Exists(list(map(lambda f: f.z3_expr(), self._vars)), self._f.z3_expr())


def mk_conjunction(l: List[Formula]):
    if len(l) == 0:
        return TruthFormula()
    elif len(l) == 1:
        return l[0]
    else:
        return ConjunctionFormula(l)


def mk_disjunction(l: List[Formula]):
    if len(l) == 0:
        return NegatedFormula(TruthFormula())
    elif len(l) == 1:
        return l[0]
    else:
        return DisjunctionFormula(l)
