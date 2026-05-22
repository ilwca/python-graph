"""
Knowledge Graph Pipeline — 10 etapas
Leitura → Validação → Normalização → Aliases → Predicados → Filtro → Dedup → Grafo → Export → Métricas
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pydantic import BaseModel, ValidationError, field_validator
from pyvis.network import Network

# ============================================================
# PATHS
# ============================================================

SAIDA_DIR = Path("/home/luca/Documentos/rc/convert-data/saida")
OUTPUT_DIR = Path("/home/luca/Documentos/rc/convert-data/grafos")
LOG_DIR   = Path("/home/luca/Documentos/rc/convert-data/logs")

# ============================================================
# CONSTANTES
# ============================================================

MIN_CONFIDENCE:    float = 0.70
MIN_ENTITY_LENGTH: int   = 3

VALID_PREDICATES: frozenset[str] = frozenset({
    "ASSOCIATED_WITH", "WORKS_FOR",         "FOUNDED",            "OWNS",
    "LOCATED_IN",      "PART_OF",           "INVESTIGATED",       "ACCUSED_OF",
    "PARTICIPATED_IN", "COMMUNICATED_WITH", "TRANSFERRED_MONEY_TO", "RESPONSIBLE_FOR",
    "RELATED_TO",      "MEMBER_OF",         "MET_WITH",           "TRAVELED_TO",
    "FINANCED",        "MANAGES",           "REPRESENTS",         "CONNECTED_TO",
})

PREDICATE_ALIASES: dict[str, str] = {
    "EMPLOYED_BY":   "WORKS_FOR",
    "WORKED_FOR":    "WORKS_FOR",
    "ASSOCIATED":    "ASSOCIATED_WITH",
    "CONNECTED":     "CONNECTED_TO",
    "TRAVELED":      "TRAVELED_TO",
    "MET":           "MET_WITH",
    "IS_MEMBER_OF":  "MEMBER_OF",
    "FUNDS":         "FINANCED",
    "MANAGED":       "MANAGES",
    "REPRESENTED":   "REPRESENTS",
    "OWNS_PROPERTY": "OWNS",
}

ENTITY_ALIASES: dict[str, str] = {
    "Epstein":                    "Jeffrey Epstein",
    "Jeff Epstein":               "Jeffrey Epstein",
    "Jeffrey E. Epstein":         "Jeffrey Epstein",
    "J. Epstein":                 "Jeffrey Epstein",
    "Maxwell":                    "Ghislaine Maxwell",
    "Ghislaine":                  "Ghislaine Maxwell",
    "G. Maxwell":                 "Ghislaine Maxwell",
    "USA":                        "United States",
    "EUA":                        "United States",
    "U.S.A.":                     "United States",
    "U.S.":                       "United States",
    "Estados Unidos":             "United States",
    "UK":                         "United Kingdom",
    "FBI":                        "Federal Bureau of Investigation",
    "DOJ":                        "Department of Justice",
    "SEC":                        "Securities and Exchange Commission",
    "IRS":                        "Internal Revenue Service",
    "SDNY":                       "Southern District of New York",
    "Little St. James":           "Little Saint James",
    "Little St James":            "Little Saint James",
}

ENTITY_BLOCKLIST: frozenset[str] = frozenset({
    "ele", "ela", "eles", "elas", "isso", "aquilo", "este", "esta",
    "estes", "estas", "aquele", "aquela", "aqueles", "aquelas",
    "empresa", "grupo", "homem", "mulher", "pessoa", "indivíduo",
    "organização", "entidade", "outro", "outra", "outros", "outras",
    "n/a", "unknown", "desconhecido", "desconhecida", "nenhum", "nenhuma",
    "algo", "alguém", "ninguém",
})

_DATE_RE   = re.compile(
    r"^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$"
    r"|^\d{4}$"
    r"|^\d{1,2}\s+de\s+\w+\s+de\s+\d{4}$",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"^\$?\d[\d,\.]*%?$")

# ============================================================
# MODELOS DE DADOS
# ============================================================

class RawRelation(BaseModel):
    subject:      str
    predicate:    str
    object:       str
    subject_type: str | None = None
    object_type:  str | None = None
    confidence:   float
    source_file:  str | None = None

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence fora de [0,1]: {v}")
        return v

    @field_validator("subject", "object", "predicate")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("campo não pode ser vazio")
        return v


@dataclass(frozen=True, slots=True)
class ProcessedRelation:
    subject:      str
    predicate:    str
    object:       str
    subject_type: str | None
    object_type:  str | None
    confidence:   float
    source_file:  str

    def edge_key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


# ============================================================
# MÉTRICAS
# ============================================================

@dataclass
class PipelineMetrics:
    files_processed:          int = 0
    files_invalid:            int = 0
    relations_loaded:         int = 0
    invalid_structure:        int = 0
    invalid_predicate:        int = 0
    low_confidence:           int = 0
    blocked_entity:           int = 0
    self_loops:               int = 0
    duplicates_removed:       int = 0
    entity_aliases_applied:   int = 0
    predicate_aliases_applied: int = 0
    relations_final:          int = 0
    nodes_final:              int = 0
    edges_final:              int = 0
    errors:           list[str]       = field(default_factory=list)
    discarded_sample: list[dict[str, Any]] = field(default_factory=list)

    def record_discard(self, data: dict, reason: str) -> None:
        if len(self.discarded_sample) < 500:
            self.discarded_sample.append({"reason": reason, **data})

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "files": {
                "processed": self.files_processed,
                "invalid":   self.files_invalid,
            },
            "relations": {
                "loaded": self.relations_loaded,
                "discarded": {
                    "invalid_structure":  self.invalid_structure,
                    "invalid_predicate":  self.invalid_predicate,
                    "low_confidence":     self.low_confidence,
                    "blocked_entity":     self.blocked_entity,
                    "self_loops":         self.self_loops,
                    "duplicates":         self.duplicates_removed,
                },
                "aliases": {
                    "entity":    self.entity_aliases_applied,
                    "predicate": self.predicate_aliases_applied,
                },
                "final": self.relations_final,
            },
            "graph": {
                "nodes": self.nodes_final,
                "edges": self.edges_final,
            },
            "errors": self.errors,
        }


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kg_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================
# ETAPA 1 — LEITURA
# ============================================================

def load_files(
    directory: Path,
    metrics:   PipelineMetrics,
    logger:    logging.Logger,
) -> Iterator[tuple[str, list[dict]]]:
    """Yields (filename, raw_list) para cada arquivo .json válido."""
    files = sorted(directory.glob("*.json"))
    if not files:
        logger.warning("Nenhum arquivo .json encontrado em %s", directory)
        return

    for path in files:
        if path.stat().st_size == 0:
            logger.warning("[SKIP] Arquivo vazio: %s", path.name)
            metrics.files_invalid += 1
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("raiz não é uma lista")
            metrics.files_processed += 1
            yield path.name, raw
        except Exception as exc:
            logger.error("[ERRO LEITURA] %s → %s", path.name, exc)
            metrics.files_invalid += 1
            metrics.errors.append(f"{path.name}: {exc}")


# ============================================================
# ETAPA 2 — VALIDAÇÃO
# ============================================================

def validate_relation(
    raw:         dict,
    source_file: str,
    metrics:     PipelineMetrics,
    logger:      logging.Logger,
) -> RawRelation | None:
    try:
        return RawRelation(source_file=source_file, **raw)
    except (ValidationError, TypeError) as exc:
        metrics.invalid_structure += 1
        metrics.record_discard(raw, f"invalid_structure: {exc}")
        logger.debug("[DESCARTADO] estrutura inválida em %s: %s", source_file, exc)
        return None


# ============================================================
# ETAPA 3 — NORMALIZAÇÃO DE ENTIDADES
# ============================================================

def normalize_entity(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x1f\x7f​‌‍﻿]", " ", text)
    text = " ".join(text.split())
    if text == text.lower() or text == text.upper():
        text = text.title()
    return text.strip()


# ============================================================
# ETAPA 4 — RESOLUÇÃO DE ALIASES DE ENTIDADES
# ============================================================

def resolve_entity(entity: str, metrics: PipelineMetrics) -> str:
    canonical = ENTITY_ALIASES.get(entity)
    if canonical:
        metrics.entity_aliases_applied += 1
        return canonical
    for alias, canon in ENTITY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", entity, re.IGNORECASE):
            if entity.lower() != canon.lower():
                metrics.entity_aliases_applied += 1
                return canon
    return entity


# ============================================================
# ETAPA 5 — NORMALIZAÇÃO DE PREDICADOS
# ============================================================

def normalize_predicate(predicate: str, metrics: PipelineMetrics) -> str | None:
    p = predicate.strip().upper()
    if p in PREDICATE_ALIASES:
        metrics.predicate_aliases_applied += 1
        p = PREDICATE_ALIASES[p]
    return p if p in VALID_PREDICATES else None


# ============================================================
# ETAPA 6 — FILTRO DE QUALIDADE
# ============================================================

def is_quality_valid(
    rel:     RawRelation,
    metrics: PipelineMetrics,
    logger:  logging.Logger,
) -> bool:
    raw = rel.model_dump()

    if rel.confidence < MIN_CONFIDENCE:
        metrics.low_confidence += 1
        metrics.record_discard(raw, f"low_confidence={rel.confidence:.2f}")
        return False

    for entity in (rel.subject, rel.object):
        if len(entity) < MIN_ENTITY_LENGTH:
            metrics.blocked_entity += 1
            metrics.record_discard(raw, f"entity_too_short: '{entity}'")
            return False
        if entity.lower() in ENTITY_BLOCKLIST:
            metrics.blocked_entity += 1
            metrics.record_discard(raw, f"blocked_entity: '{entity}'")
            logger.debug("[DESCARTADO] entidade bloqueada: '%s'", entity)
            return False
        if _DATE_RE.match(entity):
            metrics.blocked_entity += 1
            metrics.record_discard(raw, f"date_as_entity: '{entity}'")
            return False
        if _NUMBER_RE.match(entity):
            metrics.blocked_entity += 1
            metrics.record_discard(raw, f"number_as_entity: '{entity}'")
            return False

    if rel.subject == rel.object:
        metrics.self_loops += 1
        metrics.record_discard(raw, "self_loop")
        return False

    return True


# ============================================================
# ETAPA 7 — DEDUPLICAÇÃO
# ============================================================

def deduplicate(
    relations: list[ProcessedRelation],
    metrics:   PipelineMetrics,
) -> list[ProcessedRelation]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[ProcessedRelation] = []
    for rel in relations:
        key = rel.edge_key()
        if key in seen:
            metrics.duplicates_removed += 1
        else:
            seen.add(key)
            unique.append(rel)
    return unique


# ============================================================
# ETAPA 8 — CONSTRUÇÃO DO GRAFO
# ============================================================

def build_graph(relations: list[ProcessedRelation]) -> nx.MultiDiGraph:
    G: nx.MultiDiGraph = nx.MultiDiGraph()

    for rel in relations:
        if not G.has_node(rel.subject):
            G.add_node(rel.subject, entity_type=rel.subject_type or "Unknown")
        if not G.has_node(rel.object):
            G.add_node(rel.object, entity_type=rel.object_type or "Unknown")
        G.add_edge(
            rel.subject,
            rel.object,
            relation=rel.predicate,
            confidence=rel.confidence,
            source=rel.source_file,
        )

    return G


# ============================================================
# ETAPA 9 — EXPORTAÇÃO
# ============================================================

_TYPE_COLORS: dict[str, str] = {
    "Pessoa":       "#e74c3c",
    "Organização":  "#3498db",
    "Local":        "#2ecc71",
    "Avião":        "#9b59b6",
    "Documento":    "#f39c12",
    "Evento":       "#1abc9c",
    "Data":         "#95a5a6",
    "Valor":        "#e67e22",
    "Unknown":      "#7f8c8d",
}


def _node_color(entity_type: str) -> str:
    return _TYPE_COLORS.get(entity_type, _TYPE_COLORS["Unknown"])


def export_gexf(G: nx.MultiDiGraph, path: Path) -> None:
    simple = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += 1
            simple[u][v]["relations"] += f" | {data.get('relation', '')}"
        else:
            simple.add_edge(
                u, v,
                weight=1,
                relations=data.get("relation", ""),
                confidence=round(data.get("confidence", 0.0), 3),
            )
    for n, d in G.nodes(data=True):
        simple.nodes[n].update(d)
        simple.nodes[n]["degree"] = G.degree(n)
    nx.write_gexf(simple, str(path))
    print(f"[GEXF]    {path}")


def export_graphml(G: nx.MultiDiGraph, path: Path) -> None:
    simple = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if not simple.has_edge(u, v):
            simple.add_edge(u, v, **{k: str(v2) for k, v2 in data.items()})
    for n, d in G.nodes(data=True):
        simple.nodes[n].update(d)
    nx.write_graphml(simple, str(path))
    print(f"[GraphML] {path}")


def export_json(G: nx.MultiDiGraph, path: Path) -> None:
    data = {
        "nodes": [{"id": n, **d} for n, d in G.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[JSON]    {path}")


def export_html(G: nx.MultiDiGraph, path: Path) -> None:
    net = Network(
        height="950px",
        width="100%",
        directed=True,
        bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        notebook=False,
    )
    net.set_options("""{
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -10000,
          "centralGravity": 0.3,
          "springLength": 150
        },
        "stabilization": {"iterations": 250}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"type": "dynamic"},
        "font": {"size": 9, "align": "middle", "color": "#aaaaaa"}
      },
      "nodes": {"font": {"size": 11}, "borderWidth": 1.5},
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "tooltipDelay": 80
      }
    }""")

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    for node, data in G.nodes(data=True):
        deg = degrees[node]
        etype = data.get("entity_type", "Unknown")
        net.add_node(
            node,
            label=node,
            title=(
                f"<b>{node}</b><br>"
                f"Tipo: {etype}<br>"
                f"Grau: {deg} "
                f"({G.in_degree(node)} in / {G.out_degree(node)} out)"
            ),
            size=8 + 35 * (deg / max_deg),
            color=_node_color(etype),
        )

    for u, v, data in G.edges(data=True):
        net.add_edge(
            u, v,
            title=(
                f"{data.get('relation', '')} "
                f"(conf: {data.get('confidence', 0):.2f}) "
                f"[{data.get('source', '')}]"
            ),
            label=data.get("relation", ""),
        )

    net.save_graph(str(path))
    print(f"[HTML]    {path}")


def export_all(G: nx.MultiDiGraph) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_gexf(G,     OUTPUT_DIR / "graph.gexf")
    export_graphml(G,  OUTPUT_DIR / "graph.graphml")
    export_json(G,     OUTPUT_DIR / "graph.json")
    export_html(G,     OUTPUT_DIR / "graph.html")


# ============================================================
# ETAPA 10 — LOGS E MÉTRICAS
# ============================================================

def save_metrics(metrics: PipelineMetrics, logger: logging.Logger) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = metrics.to_dict()

    (LOG_DIR / "pipeline_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (LOG_DIR / "pipeline_discarded.json").write_text(
        json.dumps(metrics.discarded_sample, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_discarded = (
        metrics.invalid_structure + metrics.invalid_predicate +
        metrics.low_confidence   + metrics.blocked_entity    +
        metrics.self_loops       + metrics.duplicates_removed
    )

    logger.info("=" * 50)
    logger.info("MÉTRICAS FINAIS")
    logger.info("=" * 50)
    logger.info("Arquivos  : %d processados / %d inválidos",
                metrics.files_processed, metrics.files_invalid)
    logger.info("Relações  : %d carregadas → %d finais (%d descartadas)",
                metrics.relations_loaded, metrics.relations_final, total_discarded)
    logger.info("Descartadas breakdown:")
    logger.info("  estrutura inválida : %d", metrics.invalid_structure)
    logger.info("  predicado inválido : %d", metrics.invalid_predicate)
    logger.info("  confidence baixa   : %d", metrics.low_confidence)
    logger.info("  entidade bloqueada : %d", metrics.blocked_entity)
    logger.info("  self-loops         : %d", metrics.self_loops)
    logger.info("  duplicatas         : %d", metrics.duplicates_removed)
    logger.info("Aliases   : %d entidades / %d predicados",
                metrics.entity_aliases_applied, metrics.predicate_aliases_applied)
    logger.info("Grafo     : %d nós / %d arestas",
                metrics.nodes_final, metrics.edges_final)
    logger.info("=" * 50)


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(source_dir: Path = SAIDA_DIR) -> nx.MultiDiGraph:
    logger  = setup_logging()
    metrics = PipelineMetrics()

    logger.info("Iniciando pipeline de Knowledge Graph")
    logger.info("Fonte: %s", source_dir)

    processed: list[ProcessedRelation] = []

    # Etapas 1–7 por relação
    for filename, raw_list in load_files(source_dir, metrics, logger):
        for raw_item in raw_list:
            metrics.relations_loaded += 1

            # Etapa 2: validação
            rel = validate_relation(raw_item, filename, metrics, logger)
            if rel is None:
                continue

            # Etapa 3: normalização de entidades
            subject = normalize_entity(rel.subject)
            obj     = normalize_entity(rel.object)

            # Etapa 4: resolução de aliases
            subject = resolve_entity(subject, metrics)
            obj     = resolve_entity(obj, metrics)

            # Etapa 5: normalização de predicado
            predicate = normalize_predicate(rel.predicate, metrics)
            if predicate is None:
                metrics.invalid_predicate += 1
                metrics.record_discard(
                    rel.model_dump(),
                    f"invalid_predicate: {rel.predicate}",
                )
                logger.debug(
                    "[DESCARTADO] predicado inválido: '%s' em %s",
                    rel.predicate, filename,
                )
                continue

            # Cria relação com valores normalizados para o filtro
            rel_norm = RawRelation(
                subject=subject, predicate=predicate, object=obj,
                subject_type=rel.subject_type, object_type=rel.object_type,
                confidence=rel.confidence, source_file=filename,
            )

            # Etapa 6: filtro de qualidade
            if not is_quality_valid(rel_norm, metrics, logger):
                continue

            processed.append(ProcessedRelation(
                subject=subject, predicate=predicate, object=obj,
                subject_type=rel.subject_type, object_type=rel.object_type,
                confidence=rel.confidence, source_file=filename,
            ))

    # Etapa 7: deduplicação
    processed = deduplicate(processed, metrics)
    metrics.relations_final = len(processed)

    # Etapa 8: construção do grafo
    G = build_graph(processed)
    metrics.nodes_final = G.number_of_nodes()
    metrics.edges_final = G.number_of_edges()

    # Etapa 9: exportação
    export_all(G)

    # Etapa 10: métricas e logs
    save_metrics(metrics, logger)

    return G


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    G = run_pipeline()
    print(f"\nGrafo final: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
    print(f"Saídas : {OUTPUT_DIR}")
    print(f"Logs   : {LOG_DIR}")


if __name__ == "__main__":
    main()
