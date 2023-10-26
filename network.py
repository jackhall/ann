from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Protocol,
    Sequence,
    Type,
    TypeVar,
)
import itertools as it
import functools as ft

import networkx as nx


Downflow = TypeVar("Downflow")
Upflow = TypeVar("Upflow")
NodeID = Hashable


class Node(Protocol[Downflow, Upflow]):
    id: NodeID

    @classmethod
    def new(cls, **kwargs) -> "Node":
        ...

    def gather(self, downflows: Sequence[Downflow], upflows: Sequence[Upflow]) -> None:
        ...

    def split_downstream(self, targets: Sequence["DownstreamStateLink"]) -> list[Downflow]:
        ...

    def split_upstream(self, sources: Sequence["UpstreamStateLink"]) -> list[Upflow]:
        ...

    def adapt(self) -> None:
        ...


class SimpleLink(Protocol[Downflow, Upflow]):
    @classmethod
    def new(cls) -> "SimpleLink":
        ...

    def pull_downstream(self, node: Node) -> Downflow:
        ...

    def pull_upstream(self, node: Node) -> Upflow:
        ...


class UpstreamStateLink(
    SimpleLink[Downflow, Upflow], Protocol[Downflow, Upflow]
):
    def send_upstream(self, upflow: Upflow) -> None:
        ...


class DownstreamStateLink(SimpleLink[Downflow, Upflow], Protocol[Downflow, Upflow]):
    def send_downstream(self, downflow: Downflow) -> None:
        ...


class StatefulLink(
    UpstreamStateLink[Downflow, Upflow], 
    DownstreamStateLink[Downflow, Upflow], 
    Protocol[Downflow, Upflow]
):
    ...


@dataclass
class Network:
    node_cls: Type[Node]  # need to know: id scheme
    link_cls: Type[SimpleLink]  # need to know: whether links are stateful
    graph: nx.DiGraph
    scheduler: Callable[[nx.DiGraph], Iterable[NodeID]]

    @ft.cached_property
    def propagate(self) -> Callable[[NodeID], None]:
        Neighbors = Sequence[tuple[StatefulLink, Node]]

        def observe(n_id: NodeID) -> tuple[Neighbors, Neighbors]:
            source_pairs, target_pairs = self.links_in(n_id), self.links_out(n_id)
            self.n(n_id).gather(
                [l.pull_downstream(s) for l, s in source_pairs], 
                [l.pull_upstream(t) for l, t in target_pairs],
            )
            return source_pairs, target_pairs  # type: ignore

        if hasattr(self.link_cls, "send_downstream"):
            def stimulate_targets(n_id: NodeID) -> tuple[Neighbors, Neighbors]:
                source_pairs, target_pairs = observe(n_id)
                target_links = [p[0] for p in target_pairs]
                downflows = self.n(n_id).split_downstream(target_links)
                for link, downflow in zip(target_links, downflows):
                    link.send_downstream(downflow)
                return source_pairs, target_pairs
        else:
            stimulate_targets = observe

        if hasattr(self.link_cls, "send_upstream"):
            def stimulate_sources(n_id: NodeID) -> None:
                source_pairs, _ = stimulate_targets(n_id)
                source_links = [p[0] for p in source_pairs]
                upflows = self.n(n_id).split_upstream(source_links)
                for link, upflow in zip(source_links, upflows):
                    link.send_upstream(upflow)
        else:
            def stimulate_sources(n_id: NodeID) -> None:
                stimulate_targets(n_id)
        
        return stimulate_sources

    def l(self, s_id: NodeID, t_id: NodeID) -> SimpleLink:
        return self.graph[s_id][t_id]["obj"]

    def n(self, n_id: NodeID) -> Node:
        return self.graph.nodes[n_id]["obj"]

    def links_in(self, n_id: NodeID) -> list[tuple[SimpleLink, Node]]:
        return [
            (self.l(s_id, n_id), self.n(s_id))
            for s_id in self.graph.predecessors(n_id)
        ]

    def links_out(self, n_id: NodeID) -> list[tuple[SimpleLink, Node]]:
        return [
            (self.l(n_id, t_id), self.n(t_id))
            for t_id in self.graph.successors(n_id)
        ]

    def simulate_flow(self) -> None:
        # how can the scheduler tell when to halt?
        for n_id in self.scheduler(self.graph):
            self.propagate(n_id)

    def adapt_nodes(self) -> None:
        for _, node_attrs in self.graph.nodes.items():
            node_attrs["obj"].adapt()

    def simulation(self):
        self.simulate_flow()
        yield self

        while True:
            self.adapt_nodes()
            self.simulate_flow()
            yield self

    def connect_all(
        self,
        sources: Iterable[NodeID],
        targets: Iterable[NodeID],
        **kwargs
    ) -> None:
        self.graph.add_edges_from([
            (s, t, self.link_cls.new(**kwargs)) 
            for s, t in it.product(sources, targets)
        ])

    def add_nodes(self, kwargs_list: Iterable[dict[str, Any]]) -> None:
        for kwargs in kwargs_list:
            node = self.node_cls.new(**kwargs)
            self.graph.add_node(node.id, obj=node)

