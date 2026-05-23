"""
Exportadores do Knowledge Graph analítico.

Formatos: GEXF, GraphML, Pickle
Compatível com: Gephi, Neo4j, GraphRAG, PyVis
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

OUTPUT_DIR = Path(__file__).parent.parent / "grafos"


def _to_simple_digraph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """
    Colapsa MultiDiGraph → DiGraph.
    Arestas paralelas: acumula weight, concatena relations.
    """
    DG = nx.DiGraph()
    for n, data in G.nodes(data=True):
        # Serializa listas como strings para compatibilidade com GEXF/GraphML
        node_attrs = {}
        for k, v in data.items():
            if isinstance(v, list):
                node_attrs[k] = " | ".join(str(x) for x in v)
            else:
                node_attrs[k] = v
        DG.add_node(n, **node_attrs)

    for u, v, data in G.edges(data=True):
        if DG.has_edge(u, v):
            DG[u][v]["weight"]    += data.get("weight", 1.0)
            DG[u][v]["relations"]  = DG[u][v]["relations"] + " | " + data.get("relation", "")
        else:
            DG.add_edge(
                u, v,
                weight=data.get("weight", 1.0),
                relations=data.get("relation", ""),
                confidence=round(data.get("confidence", 0.0), 3),
                source_file=data.get("source_file", ""),
            )
    return DG


def export_gexf(G: nx.MultiDiGraph, path: Path | None = None) -> Path:
    """Exporta para GEXF (compatível com Gephi)."""
    path = path or OUTPUT_DIR / "kg_analytic.gexf"
    path.parent.mkdir(parents=True, exist_ok=True)
    DG = _to_simple_digraph(G)
    nx.write_gexf(DG, str(path))
    return path


def export_graphml(G: nx.MultiDiGraph, path: Path | None = None) -> Path:
    """Exporta para GraphML (compatível com Neo4j, yEd)."""
    path = path or OUTPUT_DIR / "kg_analytic.graphml"
    path.parent.mkdir(parents=True, exist_ok=True)
    DG = _to_simple_digraph(G)
    # GraphML exige que todos os valores sejam primitivos
    for u, v, d in DG.edges(data=True):
        for k, val in list(d.items()):
            d[k] = str(val)
    nx.write_graphml(DG, str(path))
    return path


def export_pickle(G: nx.MultiDiGraph, path: Path | None = None) -> Path:
    """
    Exporta o grafo completo (MultiDiGraph com todos os atributos) via Pickle.
    Usado como cache para recarga rápida.
    """
    path = path or OUTPUT_DIR / "kg_analytic.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: Path | None = None) -> nx.MultiDiGraph | None:
    """Carrega grafo a partir de cache Pickle. Retorna None se não existir."""
    path = path or OUTPUT_DIR / "kg_analytic.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def export_all(G: nx.MultiDiGraph, output_dir: Path | None = None) -> dict[str, Path]:
    """Exporta para todos os formatos e retorna {formato: caminho}."""
    base = output_dir or OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    return {
        "gexf":    export_gexf(G,    base / "kg_analytic.gexf"),
        "graphml": export_graphml(G, base / "kg_analytic.graphml"),
        "pickle":  export_pickle(G,  base / "kg_analytic.pkl"),
    }
