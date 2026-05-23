"""
Camada de consulta relacional e semântica sobre o Knowledge Graph.

Preparada para futura integração com LLMs / GraphRAG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class EntityInfo:
    name: str
    entity_type: str
    aliases: list[str]
    degree: int
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    eigenvector_centrality: float
    pagerank: float
    community_id: int
    in_degree: int
    out_degree: int
    neighbors: list[str] = field(default_factory=list)


@dataclass
class RelationInfo:
    source: str
    target: str
    relation: str
    confidence: float
    source_file: str
    weight: float


@dataclass
class PathInfo:
    source: str
    target: str
    path: list[str]
    length: int
    edges: list[RelationInfo] = field(default_factory=list)


class KGQuery:
    """
    Interface de consulta sobre o Knowledge Graph.

    Uso:
        q = KGQuery(G)
        info  = q.get_entity("Jeffrey Epstein")
        paths = q.shortest_path("Jeffrey Epstein", "Ghislaine Maxwell")
        ctx   = q.llm_context("Jeffrey Epstein")   # contexto para LLM
    """

    def __init__(self, G: nx.MultiDiGraph) -> None:
        self._G = G
        # DiGraph simples para algoritmos de caminho
        self._DG: nx.DiGraph = self._build_simple_digraph()

    def _build_simple_digraph(self) -> nx.DiGraph:
        DG = nx.DiGraph()
        for n, data in self._G.nodes(data=True):
            DG.add_node(n, **data)
        for u, v, data in self._G.edges(data=True):
            if DG.has_edge(u, v):
                DG[u][v]["weight"] += data.get("weight", 1.0)
            else:
                DG.add_edge(u, v, **data)
        return DG

    # ------------------------------------------------------------------
    # Busca de nó
    # ------------------------------------------------------------------

    def search_entity(self, query: str, fuzzy: bool = True) -> list[str]:
        """
        Busca nós cujo nome contenha query (case-insensitive).

        fuzzy=True: busca também nos aliases.
        """
        q = query.lower()
        matches = []
        for node, data in self._G.nodes(data=True):
            if q in node.lower():
                matches.append(node)
                continue
            if fuzzy:
                aliases = data.get("aliases", [])
                if any(q in a.lower() for a in aliases):
                    matches.append(node)
        return sorted(set(matches))

    def get_entity(self, name: str) -> EntityInfo | None:
        """Retorna metadados completos de um nó."""
        if not self._G.has_node(name):
            candidates = self.search_entity(name)
            if not candidates:
                return None
            name = candidates[0]

        data = self._G.nodes[name]
        nbrs = list(set(
            list(self._G.successors(name)) + list(self._G.predecessors(name))
        ))
        return EntityInfo(
            name=name,
            entity_type=data.get("entity_type", "Unknown"),
            aliases=data.get("aliases", []),
            degree=data.get("degree", 0),
            degree_centrality=data.get("degree_centrality", 0.0),
            betweenness_centrality=data.get("betweenness_centrality", 0.0),
            closeness_centrality=data.get("closeness_centrality", 0.0),
            eigenvector_centrality=data.get("eigenvector_centrality", 0.0),
            pagerank=data.get("pagerank", 0.0),
            community_id=data.get("community_id", -1),
            in_degree=self._G.in_degree(name),
            out_degree=self._G.out_degree(name),
            neighbors=nbrs,
        )

    # ------------------------------------------------------------------
    # Relações
    # ------------------------------------------------------------------

    def get_relations(
        self,
        entity: str,
        direction: str = "both",
    ) -> list[RelationInfo]:
        """
        Retorna todas as relações de uma entidade.

        direction: 'out' | 'in' | 'both'
        """
        if not self._G.has_node(entity):
            return []

        results: list[RelationInfo] = []

        if direction in ("out", "both"):
            for _, v, data in self._G.out_edges(entity, data=True):
                results.append(RelationInfo(
                    source=entity,
                    target=v,
                    relation=data.get("relation", ""),
                    confidence=data.get("confidence", 0.0),
                    source_file=data.get("source_file", ""),
                    weight=data.get("weight", 1.0),
                ))

        if direction in ("in", "both"):
            for u, _, data in self._G.in_edges(entity, data=True):
                results.append(RelationInfo(
                    source=u,
                    target=entity,
                    relation=data.get("relation", ""),
                    confidence=data.get("confidence", 0.0),
                    source_file=data.get("source_file", ""),
                    weight=data.get("weight", 1.0),
                ))

        return results

    def get_connections(
        self,
        entity: str,
        relation_type: str | None = None,
    ) -> list[str]:
        """Retorna entidades diretamente conectadas (em qualquer direção)."""
        if not self._G.has_node(entity):
            return []
        connected = set(self._G.successors(entity)) | set(self._G.predecessors(entity))
        if relation_type:
            rt = relation_type.upper()
            filtered = set()
            for _, v, data in self._G.out_edges(entity, data=True):
                if data.get("relation") == rt:
                    filtered.add(v)
            for u, _, data in self._G.in_edges(entity, data=True):
                if data.get("relation") == rt:
                    filtered.add(u)
            return sorted(filtered)
        return sorted(connected)

    # ------------------------------------------------------------------
    # Caminhos
    # ------------------------------------------------------------------

    def shortest_path(self, source: str, target: str) -> PathInfo | None:
        """Retorna o caminho mais curto entre dois nós (direcionado)."""
        if not (self._G.has_node(source) and self._G.has_node(target)):
            return None
        try:
            path = nx.shortest_path(self._DG, source, target)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

        edges = self._path_edges(path)
        return PathInfo(source=source, target=target, path=path, length=len(path) - 1, edges=edges)

    def all_simple_paths(
        self,
        source: str,
        target: str,
        cutoff: int = 5,
    ) -> list[PathInfo]:
        """Retorna todos os caminhos simples até comprimento cutoff."""
        if not (self._G.has_node(source) and self._G.has_node(target)):
            return []
        paths = []
        for path in nx.all_simple_paths(self._DG, source, target, cutoff=cutoff):
            edges = self._path_edges(path)
            paths.append(PathInfo(
                source=source, target=target,
                path=path, length=len(path) - 1, edges=edges,
            ))
        return paths

    def _path_edges(self, path: list[str]) -> list[RelationInfo]:
        edges = []
        for u, v in zip(path, path[1:]):
            # Pega a primeira aresta entre u→v no MultiDiGraph
            edge_data = next(iter(self._G[u][v].values()), {}) if self._G.has_edge(u, v) else {}
            edges.append(RelationInfo(
                source=u,
                target=v,
                relation=edge_data.get("relation", ""),
                confidence=edge_data.get("confidence", 0.0),
                source_file=edge_data.get("source_file", ""),
                weight=edge_data.get("weight", 1.0),
            ))
        return edges

    # ------------------------------------------------------------------
    # Ego graph / vizinhança
    # ------------------------------------------------------------------

    def ego_graph(self, entity: str, radius: int = 1) -> nx.MultiDiGraph:
        """Retorna o subgrafo centrado em entity com raio radius."""
        if not self._G.has_node(entity):
            return nx.MultiDiGraph()
        return nx.ego_graph(self._G, entity, radius=radius, undirected=True)

    def neighbors(self, entity: str) -> dict[str, list[str]]:
        """Retorna predecessores e sucessores diretos."""
        if not self._G.has_node(entity):
            return {"predecessors": [], "successors": []}
        return {
            "predecessors": sorted(self._G.predecessors(entity)),
            "successors":   sorted(self._G.successors(entity)),
        }

    def descendants(self, entity: str) -> list[str]:
        """Todos os nós alcançáveis a partir de entity (direcionado)."""
        if not self._G.has_node(entity):
            return []
        return sorted(nx.descendants(self._DG, entity))

    def ancestors(self, entity: str) -> list[str]:
        """Todos os nós que alcançam entity (direcionado)."""
        if not self._G.has_node(entity):
            return []
        return sorted(nx.ancestors(self._DG, entity))

    # ------------------------------------------------------------------
    # Comunidades
    # ------------------------------------------------------------------

    def get_community(self, entity: str) -> list[str]:
        """Retorna todos os membros da comunidade da entidade."""
        if not self._G.has_node(entity):
            return []
        cid = self._G.nodes[entity].get("community_id", -1)
        return sorted(
            n for n, d in self._G.nodes(data=True)
            if d.get("community_id") == cid
        )

    def search_communities(self, query: str) -> list[dict[str, Any]]:
        """Busca comunidades que contenham entidades matching query."""
        matching = self.search_entity(query)
        result: dict[int, dict] = {}
        for node in matching:
            cid = self._G.nodes[node].get("community_id", -1)
            if cid not in result:
                members = self.get_community(node)
                result[cid] = {"community_id": cid, "size": len(members), "matching": [], "members": members}
            result[cid]["matching"].append(node)
        return list(result.values())

    # ------------------------------------------------------------------
    # Nós centrais
    # ------------------------------------------------------------------

    def central_nodes(
        self,
        metric: str = "pagerank",
        n: int = 10,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retorna os n nós mais centrais segundo metric.

        metric: 'pagerank' | 'degree_centrality' | 'betweenness_centrality'
                | 'closeness_centrality' | 'eigenvector_centrality'
        """
        nodes = (
            (node, data) for node, data in self._G.nodes(data=True)
            if entity_type is None or data.get("entity_type") == entity_type
        )
        ranked = sorted(
            nodes,
            key=lambda nd: nd[1].get(metric, 0.0),
            reverse=True,
        )[:n]
        return [
            {
                "node": node,
                "entity_type": data.get("entity_type"),
                metric: data.get(metric, 0.0),
                "community_id": data.get("community_id", -1),
            }
            for node, data in ranked
        ]

    # ------------------------------------------------------------------
    # Contexto para LLM (GraphRAG)
    # ------------------------------------------------------------------

    def llm_context(
        self,
        entity: str,
        radius: int = 2,
        max_paths: int = 3,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """
        Gera contexto estrutural enriquecido para uso por uma LLM.

        Retorna dict com:
          - entity_info: metadados da entidade
          - direct_relations: relações de primeiro grau
          - community_members: membros da mesma comunidade
          - ego_nodes: nós no raio ego_graph
          - central_in_community: mais centrais da comunidade
          - structural_summary: resumo textual
        """
        entity_info = self.get_entity(entity)
        if entity_info is None:
            return {"error": f"Entidade '{entity}' não encontrada no grafo."}

        direct_relations = self.get_relations(entity)
        community_members = self.get_community(entity)

        ego = self.ego_graph(entity, radius=radius)
        ego_nodes = [n for n in ego.nodes() if n != entity]

        central_in_community = sorted(
            community_members,
            key=lambda n: self._G.nodes[n].get("pagerank", 0.0),
            reverse=True,
        )[:top_n]

        out_rels = [(r.relation, r.target) for r in direct_relations if r.source == entity][:10]
        in_rels  = [(r.relation, r.source) for r in direct_relations if r.target == entity][:10]

        summary_lines = [
            f"Entidade: {entity} ({entity_info.entity_type})",
            f"Grau: {entity_info.degree} | PageRank: {entity_info.pagerank:.4f}",
            f"Betweenness: {entity_info.betweenness_centrality:.4f} | Comunidade: {entity_info.community_id}",
            f"Relações de saída ({len(out_rels)}): " + ", ".join(f"{r}→{t}" for r, t in out_rels),
            f"Relações de entrada ({len(in_rels)}): " + ", ".join(f"{s}→{r}" for r, s in in_rels),
            f"Membros da comunidade: {len(community_members)}",
        ]

        return {
            "entity": entity,
            "entity_info": {
                "entity_type":             entity_info.entity_type,
                "aliases":                 entity_info.aliases,
                "degree":                  entity_info.degree,
                "in_degree":               entity_info.in_degree,
                "out_degree":              entity_info.out_degree,
                "pagerank":                entity_info.pagerank,
                "betweenness_centrality":  entity_info.betweenness_centrality,
                "closeness_centrality":    entity_info.closeness_centrality,
                "eigenvector_centrality":  entity_info.eigenvector_centrality,
                "community_id":            entity_info.community_id,
            },
            "direct_relations": [
                {
                    "source":     r.source,
                    "relation":   r.relation,
                    "target":     r.target,
                    "confidence": r.confidence,
                }
                for r in direct_relations
            ],
            "community_members":      community_members,
            "ego_nodes":              ego_nodes,
            "central_in_community":   central_in_community,
            "structural_summary":     "\n".join(summary_lines),
        }
