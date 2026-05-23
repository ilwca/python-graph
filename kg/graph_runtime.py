"""
Orquestrador principal da camada analítica do Knowledge Graph.

Fluxo:
  1. Carrega grafo (JSON → MultiDiGraph, ou Pickle cache)
  2. Calcula métricas de centralidade
  3. Detecta comunidades (Louvain)
  4. Exporta (GEXF, GraphML, Pickle)
  5. Disponibiliza interface de consulta (KGQuery)

Uso rápido:
    from kg import KGRuntime
    rt = KGRuntime()          # carrega, calcula, exporta
    q  = rt.query             # interface de consulta
    q.get_entity("Jeffrey Epstein")
    q.shortest_path("Jeffrey Epstein", "Ghislaine Maxwell")
    q.llm_context("Jeffrey Epstein")
"""
from __future__ import annotations

import time
from pathlib import Path

import networkx as nx

from .communities import community_summary, detect_communities
from .exporters import export_all, load_pickle
from .graph_builder import SAIDA_DIR, build_graph
from .metrics import compute_metrics, top_nodes
from .queries import KGQuery


class KGRuntime:
    """
    Ponto central de acesso ao Knowledge Graph analítico.

    Parâmetros:
        source_dir    : diretório com JSONs processados
        use_cache     : tenta carregar Pickle antes de reconstruir
        force_rebuild : ignora cache e reconstrói sempre
        export        : salva GEXF/GraphML/Pickle após construção
        verbose       : imprime progresso
    """

    def __init__(
        self,
        source_dir: Path = SAIDA_DIR,
        use_cache: bool = True,
        force_rebuild: bool = False,
        export: bool = True,
        verbose: bool = True,
    ) -> None:
        self._verbose = verbose
        self.graph: nx.MultiDiGraph = self._load(
            source_dir, use_cache, force_rebuild, export
        )
        self.query = KGQuery(self.graph)

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------

    def _load(
        self,
        source_dir: Path,
        use_cache: bool,
        force_rebuild: bool,
        do_export: bool,
    ) -> nx.MultiDiGraph:
        if use_cache and not force_rebuild:
            G = load_pickle()
            if G is not None:
                self._log(
                    f"[cache] Grafo carregado do Pickle: "
                    f"{G.number_of_nodes()} nós / {G.number_of_edges()} arestas"
                )
                return G

        self._log(f"[build] Construindo grafo a partir de {source_dir} ...")
        t0 = time.perf_counter()

        G = build_graph(source_dir)
        self._log(
            f"[build] Grafo construído: "
            f"{G.number_of_nodes()} nós / {G.number_of_edges()} arestas "
            f"({time.perf_counter() - t0:.2f}s)"
        )

        self._log("[metrics] Calculando centralidades ...")
        t1 = time.perf_counter()
        compute_metrics(G)
        self._log(f"[metrics] Concluído ({time.perf_counter() - t1:.2f}s)")

        self._log("[communities] Detectando comunidades (Louvain) ...")
        t2 = time.perf_counter()
        detect_communities(G)
        n_communities = len({d.get("community_id") for _, d in G.nodes(data=True)})
        self._log(
            f"[communities] {n_communities} comunidades detectadas "
            f"({time.perf_counter() - t2:.2f}s)"
        )

        if do_export:
            self._log("[export] Exportando grafos ...")
            t3 = time.perf_counter()
            paths = export_all(G)
            for fmt, path in paths.items():
                self._log(f"[export]   {fmt.upper()}: {path}")
            self._log(f"[export] Concluído ({time.perf_counter() - t3:.2f}s)")

        self._log(
            f"[runtime] Pipeline completo em {time.perf_counter() - t0:.2f}s"
        )
        return G

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Atalhos de análise
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Resumo geral do grafo: nós, arestas, comunidades, top hubs."""
        G = self.graph
        communities = {d.get("community_id") for _, d in G.nodes(data=True)}
        return {
            "nodes":       G.number_of_nodes(),
            "edges":       G.number_of_edges(),
            "communities": len(communities),
            "top_pagerank":      top_nodes(G, "pagerank",              10),
            "top_betweenness":   top_nodes(G, "betweenness_centrality", 10),
            "top_degree":        top_nodes(G, "degree_centrality",      10),
            "community_summary": community_summary(G),
        }

    def print_summary(self) -> None:
        """Imprime resumo formatado no terminal."""
        s = self.summary()
        print("\n" + "=" * 60)
        print(f"  Knowledge Graph — Resumo Analítico")
        print("=" * 60)
        print(f"  Nós        : {s['nodes']}")
        print(f"  Arestas    : {s['edges']}")
        print(f"  Comunidades: {s['communities']}")

        print("\n  Top 10 — PageRank (entidades mais influentes):")
        for i, (node, score) in enumerate(s["top_pagerank"], 1):
            print(f"    {i:>2}. {node:<45} {score:.6f}")

        print("\n  Top 10 — Betweenness (intermediários estratégicos):")
        for i, (node, score) in enumerate(s["top_betweenness"], 1):
            print(f"    {i:>2}. {node:<45} {score:.6f}")

        print("\n  Maiores comunidades:")
        for c in s["community_summary"][:5]:
            tops = ", ".join(c["top_nodes"])
            print(f"    Comunidade {c['community_id']:>3}: {c['size']:>4} membros — {tops}")
        print("=" * 60 + "\n")
