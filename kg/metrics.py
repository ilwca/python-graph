"""
Calcula métricas de rede e adiciona como atributos nos nós do grafo.

Centralidades calculadas:
  degree_centrality, betweenness_centrality, closeness_centrality,
  eigenvector_centrality, pagerank
"""
from __future__ import annotations

import networkx as nx


def _to_digraph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Colapsa MultiDiGraph → DiGraph somando pesos de arestas paralelas."""
    DG = nx.DiGraph()
    for n, data in G.nodes(data=True):
        DG.add_node(n, **data)
    for u, v, data in G.edges(data=True):
        if DG.has_edge(u, v):
            DG[u][v]["weight"] += data.get("weight", 1.0)
        else:
            DG.add_edge(u, v, weight=data.get("weight", 1.0))
    return DG


def compute_metrics(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Calcula todas as métricas de centralidade e insere como atributos nos nós.

    Retorna o mesmo grafo G com os atributos atualizados.
    """
    if G.number_of_nodes() == 0:
        return G

    DG = _to_digraph(G)

    degree_cent      = nx.degree_centrality(DG)
    betweenness_cent = nx.betweenness_centrality(DG, normalized=True, weight="weight")
    closeness_cent   = nx.closeness_centrality(DG)
    pagerank         = nx.pagerank(DG, weight="weight", max_iter=500)

    try:
        eigenvector_cent = nx.eigenvector_centrality(DG, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        # Fallback para grafos desconexos ou com convergência difícil
        eigenvector_cent = {n: 0.0 for n in DG.nodes()}

    for node in G.nodes():
        G.nodes[node]["degree_centrality"]      = round(degree_cent.get(node, 0.0),      6)
        G.nodes[node]["betweenness_centrality"] = round(betweenness_cent.get(node, 0.0), 6)
        G.nodes[node]["closeness_centrality"]   = round(closeness_cent.get(node, 0.0),   6)
        G.nodes[node]["eigenvector_centrality"] = round(eigenvector_cent.get(node, 0.0), 6)
        G.nodes[node]["pagerank"]               = round(pagerank.get(node, 0.0),          6)

    return G


def top_nodes(
    G: nx.MultiDiGraph,
    metric: str = "pagerank",
    n: int = 10,
) -> list[tuple[str, float]]:
    """
    Retorna os n nós com maior valor em uma métrica de centralidade.

    metric: 'pagerank' | 'degree_centrality' | 'betweenness_centrality'
            | 'closeness_centrality' | 'eigenvector_centrality'
    """
    scores = {
        node: data.get(metric, 0.0)
        for node, data in G.nodes(data=True)
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
