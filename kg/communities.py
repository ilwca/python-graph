"""
Detecção de comunidades via algoritmo de Louvain.

Salva community_id como atributo em cada nó.
"""
from __future__ import annotations

import community as community_louvain
import networkx as nx


def detect_communities(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Executa Louvain sobre versão não-direcionada do grafo.
    Salva community_id em cada nó de G.

    Retorna G com atributos atualizados.
    """
    if G.number_of_nodes() == 0:
        return G

    # Louvain exige grafo não-direcionado
    UG = nx.Graph()
    for n in G.nodes():
        UG.add_node(n)
    for u, v, data in G.edges(data=True):
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += data.get("weight", 1.0)
        else:
            UG.add_edge(u, v, weight=data.get("weight", 1.0))

    partition: dict[str, int] = community_louvain.best_partition(UG, weight="weight")

    for node in G.nodes():
        G.nodes[node]["community_id"] = partition.get(node, -1)

    return G


def get_community_members(G: nx.MultiDiGraph) -> dict[int, list[str]]:
    """Retorna {community_id: [nó, ...]} para todos os nós do grafo."""
    communities: dict[int, list[str]] = {}
    for node, data in G.nodes(data=True):
        cid = data.get("community_id", -1)
        communities.setdefault(cid, []).append(node)
    return {cid: sorted(members) for cid, members in sorted(communities.items())}


def community_summary(G: nx.MultiDiGraph) -> list[dict]:
    """
    Retorna lista de resumos por comunidade, ordenada por tamanho decrescente.
    Cada item: {community_id, size, top_nodes_by_pagerank}
    """
    members = get_community_members(G)
    summary = []
    for cid, nodes in members.items():
        top = sorted(
            nodes,
            key=lambda n: G.nodes[n].get("pagerank", 0.0),
            reverse=True,
        )[:5]
        summary.append({
            "community_id": cid,
            "size": len(nodes),
            "top_nodes": top,
        })
    return sorted(summary, key=lambda x: x["size"], reverse=True)
