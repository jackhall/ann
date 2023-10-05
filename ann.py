from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import (
    Callable,
    ClassVar,
    Optional,
    Sequence,
)

import networkx as nx
import bitstring as bs

from network import Node, UpstreamStateLink


@contextmanager
def random_seed(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


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


Sensed = list[float]  # use numpy


@dataclass
class Dreamed:
    bits: bs.Bits  # use numpy
    paid: float


@dataclass
class Neuron(Node[Sensed, Dreamed]):
    id: str
    bias: float
    activation: bs.Bits = bs.Bits()
    dreamed: Dreamed = Dreamed(bs.Bits(), 0.0)

    @classmethod
    def new(cls, id_: str, bias: float) -> "Neuron":
        return cls(id_, bias, bs.Bits())

    def gather(self, downflows: Sequence[Sensed], upflows: Sequence[Dreamed]) -> None:
        activation = 

    def split_upstream(self, links: Sequence["Synapse"]) -> list[Dreamed]:
        ...

    def split_downstream(self, targets) -> list:
        # leave undefined
        ...

    def adapt(self) -> None:
        ...


@dataclass
class Synapse(UpstreamStateLink[Sensed, Dreamed]):
    weight: float
    dreamed: Dreamed = Dreamed(bs.Bits(), 0.0)

    @classmethod
    def new(
        cls,
        weight_factory: Optional[Callable[[], float]] = None,
    ) -> "Synapse":
        if weight_factory is None:
            weight_factory = lambda: random.uniform(-1, 1)
        return cls(weight_factory())

    def pull_downstream(self, node: Neuron) -> Sensed:
        return [int(bit) * self.weight for bit in node.activation]

    def pull_upstream(self, node: Node) -> Dreamed:
        ...

    def send_upstream(self, upflow: Dreamed) -> None:
        ...


def perceptron_scheduler(graph: nx.DiGraph):
    # iterate downstream and then back upstream
    ...


def build_xor() -> NeuralNetwork:
    return NeuralNetwork.perceptron([2, 2, 1, 1], lambda: random.uniform(-1, 1))

