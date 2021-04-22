from __future__ import annotations

from itertools import chain
from typing import Dict, Set, Tuple, Iterable


class Place:

    def __init__(self, name: str, t: ComponentType):
        self._name: str = name
        self._type: ComponentType = t
        self._bound_ports: Dict[str, Port] = {}

    def name(self) -> str:
        return self._name

    def type(self) -> ComponentType:
        return self._type

    def add_bound_port(self, p: Port) -> None:
        assert p.type() == self._type
        assert p.name() not in self._bound_ports
        self._bound_ports[p.name()] = p

    def bound_ports(self) -> Iterable[Port]:
        return self._bound_ports.values()

    def bound_use_ports(self) -> Iterable[Port]:
        return (p for p in self._bound_ports.values() if p.is_use_port())

    def bound_provide_ports(self) -> Iterable[Port]:
        return (p for p in self._bound_ports.values() if p.is_provide_port())


class Port:

    def __init__(self, name: str, t: ComponentType, provide_port: bool, bound_places: Iterable[Place]):
        self._name: str = name
        self._type: ComponentType = t
        self._provide_port: bool = provide_port
        self._bound_places: Set[Place] = set(bound_places)
        for p in bound_places:
            p.add_bound_port(self)

    def name(self) -> str:
        return self._name

    def type(self) -> ComponentType:
        return self._type

    def is_provide_port(self) -> bool:
        return self._provide_port

    def is_use_port(self) -> bool:
        return not self._provide_port

    def is_active(self, active_places: Set[Place]):
        return any(pl in self._bound_places for pl in active_places)

    def bound_places(self) -> Iterable[Place]:
        return self._bound_places

    def is_bound_to(self, p: Place) -> bool:
        assert p in self._type.places()
        return p in self._bound_places


class Transition:

    def __init__(self, src: Place, dst: Place, dock: int = 0):
        assert src.type() == dst.type()
        self._src = src
        self._dst = dst
        self._dock = dock

    def source(self) -> Place:
        return self._src

    def destination(self) -> Tuple[Place, int]:
        return self._dst, self._dock

    def activating_ports(self) -> Iterable[Port]:
        t = self._src.type()
        return (p for p in t.ports() if p.is_bound_to(self._dst) and not p.is_bound_to(self._src))

    def deactivating_ports(self) -> Iterable[Port]:
        t = self._src.type()
        # a port that satsifies the condition could remain active after firing the transition, if there are other
        # active places bound to that port
        return (p for p in t.ports() if p.is_bound_to(self._src) and not p.is_bound_to(self._dst))


class Behavior:

    def __init__(self, name: str, t: ComponentType):
        self._name: str = name
        self._type: ComponentType = t
        self._transitions: Dict[str, Transition] = {}

    def name(self) -> str:
        return self._name

    def type(self) -> ComponentType:
        return self._type

    def add_transition(self, name: str, src: Place, dst: Place, dock: int = 0):
        assert name not in self._transitions
        assert src in self._type.places()
        assert dst in self._type.places()
        assert not self._introduces_cycle(src, dst)
        self._transitions[name] = Transition(src, dst, dock)

    def transitions(self) -> Iterable[Transition]:
        return self._transitions.values()

    def transitions_from(self, src: Place) -> Iterable[Transition]:
        assert src in self._type.places()
        return (tr for tr in self._transitions.values() if tr.source() == src)

    def transitions_to(self, dst: Place, dock: int = 0) -> Iterable[Transition]:
        assert dst in self._type.places()
        return (tr for tr in self._transitions.values() if tr.destination() == (dst, dock))

    def has_transitions_from(self, active_places: Set[Place]) -> bool:
        return any(tr.source() in active_places for tr in self._transitions.values())

    def is_final_place(self, p: Place) -> bool:
        return not any(tr.source() == p for tr in self._transitions.values())

    def _introduces_cycle(self, new_tr_src: Place, new_tr_dst: Place):
        reached: Set[Place] = {new_tr_dst}
        while reached:
            p = reached.pop()
            if p == new_tr_src:
                return True
            for t in self._transitions.values():
                if t.source() == p:
                    reached.add(t.destination()[0])
        return False


class ComponentType:

    def __init__(self, name: str):
        self._name: str = name
        self._behaviors: Dict[name, Behavior] = {}
        self._provide_ports: Dict[str, Port] = {}
        self._use_ports: Dict[str, Port] = {}
        self._places: Dict[str, Place] = {}

    def add_behavior(self, name: str) -> Behavior:
        assert name not in self._behaviors
        b = Behavior(name, self)
        self._behaviors[name] = b
        return b

    def add_place(self, name: str) -> Place:
        assert name not in self._places
        p = Place(name, self)
        self._places[name] = p
        return p

    def get_place(self, name: str) -> Place:
        assert name in self._places
        return self._places[name]

    def add_provide_port(self, name: str, bound_places: Iterable[Place]) -> Port:
        assert name not in self._use_ports and name not in self._provide_ports
        assert all(p in self._places.values() for p in bound_places)
        p = Port(name, self, True, bound_places)
        self._provide_ports[name] = p
        return p

    def add_use_port(self, name: str, bound_places: Iterable[Place]) -> Port:
        assert name not in self._provide_ports and name not in self._use_ports
        assert all(p in self._places.values() for p in bound_places)
        p = Port(name, self, False, bound_places)
        self._use_ports[name] = p
        return p

    def provide_ports(self) -> Iterable[Port]:
        return self._provide_ports.values()

    def use_ports(self) -> Iterable[Port]:
        return self._use_ports.values()

    def ports(self) -> Iterable[Port]:
        return chain(self._provide_ports.values(), self._use_ports.values())

    def get_provide_port(self, name: str) -> Port:
        assert name in self._provide_ports
        return self._provide_ports[name]

    def get_use_port(self, name: str) -> Port:
        assert name in self._use_ports
        return self._use_ports[name]

    def get_port(self, name: str) -> Port:
        if name in self._provide_ports:
            return self._provide_ports[name]
        if name in self._use_ports:
            return self._use_ports[name]
        assert False

    def get_behavior(self, name: str) -> Behavior:
        assert name in self._behaviors
        return self._behaviors[name]

    def behaviors(self) -> Iterable[Behavior]:
        return self._behaviors.values()

    def places(self) -> Iterable[Place]:
        return self._places.values()


class ComponentInstance:

    def __init__(self, identifier: str, t: ComponentType):
        self._id: str = identifier
        self._type: ComponentType = t
        self._connections: Dict[Port, Set[Tuple[ComponentInstance, Port]]] = {p: set() for p in self._type.ports()}

    def id(self) -> str:
        return self._id

    def type(self) -> ComponentType:
        return self._type

    def connections(self, p: Port) -> Iterable[Tuple[ComponentInstance, Port]]:
        assert p in self._connections
        return self._connections[p]

    def is_connected(self, p: Port) -> bool:
        assert p in self._type.ports()
        return len(self._connections[p]) > 0

    def connect_provide_port(self, provide_port: Port, user: ComponentInstance, use_port: Port):
        assert provide_port in self._type.provide_ports()
        assert use_port in user._type.use_ports()
        self._connections[provide_port].add((user, use_port))

    def connect_use_port(self, use_port: Port, provider: ComponentInstance, provide_port: Port):
        assert use_port in self._type.use_ports()
        assert provide_port in provider._type.provide_ports()
        # a use port should only be connected to one provider
        assert not self._connections[use_port]
        self._connections[use_port].add((provider, provide_port))


class Assembly:

    def __init__(self):
        self._instances: Dict[str, ComponentInstance] = {}

    def add_instance(self, instance_id: str, t: ComponentType) -> ComponentInstance:
        assert instance_id not in self._instances
        c = ComponentInstance(instance_id, t)
        self._instances[instance_id] = c
        return c

    def connect_instances(self,
                          provider: ComponentInstance,
                          provide_port: Port,
                          user: ComponentInstance,
                          use_port: Port):
        assert provider in self._instances.values()
        assert provide_port.is_provide_port()
        assert user in self._instances.values()
        assert use_port.is_use_port()
        provider.connect_provide_port(provide_port, user, use_port)
        user.connect_use_port(use_port, provider, provide_port)

    def connect_instances_id(self,
                             provider_id: str,
                             provide_port_name: str,
                             user_id: str,
                             use_port_name: str):
        provider = self.get_instance(provider_id)
        user = self.get_instance(user_id)
        provide_port = provider.type().get_provide_port(provide_port_name)
        use_port = user.type().get_use_port(use_port_name)
        self.connect_instances(provider, provide_port, user, use_port)

    def get_instance(self, identifier: str) -> ComponentInstance:
        assert identifier in self._instances
        return self._instances[identifier]

    def instances(self) -> Iterable[ComponentInstance]:
        return self._instances.values()
