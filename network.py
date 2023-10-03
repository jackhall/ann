from dataclasses import dataclass
from contextlib import contextmanager
import random
from typing import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Optional,
    Protocol,
    Type,
    TypeVar,
)
import itertools as it

import networkx as nx
import bitstring as bs


@contextmanager
def random_seed(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


U = TypeVar("U")
D = TypeVar("D")


class Node(Protocol[U, D, A]):
    @classmethod
    def new(cls) -> "Node":
        ...

    def gather(self, downflows: list[D], upflows: list[U]) -> A:
        ...

    def propagate_downstream(self, act: A, targets: Collection["Link"]) -> list[D]:
        ...

    def propagate_upstream(self, act: A, sources: Collection["Link"]) -> list[U]:
        ...

    def update(self):
        ...


class Link(Protocol[U, D]):
    @classmethod
    def new(cls) -> "Link":
        ...

    def pull_downstream(self, node: Node) -> D:
        ...

    def pull_upstream(self, node: Node) -> U:
        ...


class UpstreamStateLink(Link[U, D], Protocol[U, D]):
    def send_upstream(self, upflow: U) -> None:
        ...


class DownstreamStateLink(Link[U, D], Protocol[U, D]):
    def send_downstream(self, downflow: D) -> None:
        ...


class StatefulLink(UpstreamStateLink[U, D], DownstreamStateLink[U, D], Protocol[U, D]):
    ...


@dataclass
class Network:
    node_cls: Type[Node]  # need to know: id scheme
    link_cls: Type[Link]  # need to know: whether links are stateful
    graph: nx.DiGraph
    scheduler: Callable[[nx.DiGraph], Iterable[Hashable]]

    def l(self, s_id: Hashable, t_id: Hashable) -> Link:
        return self.graph[s_id][t_id]["obj"]

    def n(self, n_id: Hashable) -> Node:
        return self.graph.nodes[n_id]["obj"]

    def links_in(self, n_id: Hashable) -> list[tuple[Link, Node]]:
        return [
            (self.l(s_id, n_id), self.n(s_id))
            for s_id in self.graph.predecessors(n_id)
        ]

    def links_out(self, n_id: Hashable) -> list[tuple[Link, Node]]:
        return [
            (self.l(n_id, t_id), self.n(t_id))
            for t_id in self.graph.successors(n_id)
        ]

    def simulate_flow(self):
        # how can the scheduler tell when to halt?
        for n_id in self.scheduler(self.graph):
            source_pairs = self.links_in(n_id)
            target_pairs = self.links_out(n_id)

            node = self.n(n_id)
            activation = node.gather(
                [l.pull_downstream(s) for l, s in source_pairs], 
                [l.pull_upstream(t) for l, t in target_pairs],
            )

            # get these conditionals out of the loop
            if hasattr(self.link_cls, "send_downstream"):
                source_links = [pair[0] for pair in source_pairs]
                node.propagate_downstream(activation, source_links)

            if hasattr(self.link_cls, "send_upstream"):
                target_links = [pair[0] for pair in source_pairs]
                node.propagate_upstream(activation, target_links)

    def update_nodes(self):
        for _, node_attrs in self.graph.nodes.items():
            node_attrs["obj"].update()

    def simulation(self):
        self.simulate_flow()
        yield self

        while True:
            self.update_nodes()
            self.simulate_flow()
            yield self

    def connect_all(self, sources: Iterable[Node], targets: Iterable[Node], **kwargs) -> None:
        self.graph.add_edges_from([
            (s, t, self.link_cls.new(**kwargs)) 
            for s, t in it.product(sources, targets)
        ])


@dataclass
class NeuralNetwork:
    graph: nx.DiGraph
    layers: dict[str, int]

    @staticmethod
    def node_ids(layer, n) -> list[str]:
        return [f"{layer}-{i}" for i in range(n)]

    @classmethod
    def empty(cls, num_inputs, num_outputs) -> "NeuralNetwork":
        network = nx.DiGraph()
        network.add_nodes_from(
            cls.node_ids("in", num_inputs) 
            + cls.node_ids("out", num_outputs)
        )
        return cls(network, {"in": num_inputs, "out": num_outputs})

    @classmethod
    def perceptron(
        cls,
        layer_counts: list[int],
        weight_factory: Callable[[], float]
    ) -> "NeuralNetwork":
        assert len(layer_counts) > 2, "no hidden layers"
        self = cls.empty(layer_counts[0], layer_counts[-1])

        source_ids = cls.node_ids("in", layer_counts[0])

        for layer, n in enumerate(layer_counts[1:-1], start=1):
            node_ids = cls.node_ids(layer, n)
            node_objs = [Neuron.random() for _ in range(n)]
            self.graph.add_nodes_from(zip(node_ids, node_objs))
            self.connect_all(source_ids, node_ids, weight_factory=weight_factory)

            source_ids = node_ids
            self.layers[str(layer)] = n
        
        return self

    def connect_all(self, sources: Iterable[Node], targets: Iterable[Node], **kwargs) -> None:
        link_ids = it.product(sources, targets)
        link_tuples = [
            (s, t, Link.random(s, t, **kwargs)) for s, t in link_ids
        ]
        self.graph.add_edges_from(link_tuples)


@dataclass
class Synapse:
    weight: float
    paid: float

    @classmethod
    def new(
        cls,
        weight_factory: Optional[Callable[[], float]] = None,
    ) -> "Synapse":
        if weight_factory is None:
            weight_factory = lambda: random.uniform(-1, 1)
        return cls(weight_factory(), 0.0)


@dataclass
class Neuron:
    bias: float
    sensed: bs.Bits
    dreamed: bs.Bits
    account: float

    @classmethod
    def new(cls, bias_factory: Optional[Callable[[], float]] = None) -> "Neuron":
        if bias_factory is None:
            bias_factory = lambda: random.uniform(-1, 1)
        return cls(bias_factory(), bs.Bits(), bs.Bits(), 0.0)


def build_xor() -> NeuralNetwork:
    return NeuralNetwork.perceptron([2, 2, 1, 1], lambda: random.uniform(-1, 1))

