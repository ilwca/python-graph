"""
Constrói um nx.MultiDiGraph a partir dos arquivos JSON já tratados pelo pipeline.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import networkx as nx

SAIDA_DIR = Path(__file__).parent.parent / "saida"

MIN_CONFIDENCE: float = 0.70
MIN_ENTITY_LENGTH: int = 3

VALID_PREDICATES: frozenset[str] = frozenset({
    "ASSOCIATED_WITH", "WORKS_FOR",          "FOUNDED",             "OWNS",
    "LOCATED_IN",      "PART_OF",            "INVESTIGATED",        "ACCUSED_OF",
    "PARTICIPATED_IN", "COMMUNICATED_WITH",  "TRANSFERRED_MONEY_TO","RESPONSIBLE_FOR",
    "RELATED_TO",      "MEMBER_OF",          "MET_WITH",            "TRAVELED_TO",
    "FINANCED",        "MANAGES",            "REPRESENTS",          "CONNECTED_TO",
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
    "Epstein":               "Jeffrey Epstein",
    "Jeff Epstein":          "Jeffrey Epstein",
    "Jeffrey E. Epstein":    "Jeffrey Epstein",
    "J. Epstein":            "Jeffrey Epstein",
    "Maxwell":               "Ghislaine Maxwell",
    "Ghislaine":             "Ghislaine Maxwell",
    "G. Maxwell":            "Ghislaine Maxwell",
    "USA":                   "United States",
    "EUA":                   "United States",
    "U.S.A.":                "United States",
    "U.S.":                  "United States",
    "Estados Unidos":        "United States",
    "UK":                    "United Kingdom",
    "FBI":                   "Federal Bureau of Investigation",
    "DOJ":                   "Department of Justice",
    "SEC":                   "Securities and Exchange Commission",
    "IRS":                   "Internal Revenue Service",
    "SDNY":                  "Southern District of New York",
    "Little St. James":      "Little Saint James",
    "Little St James":       "Little Saint James",
}

ENTITY_BLOCKLIST: frozenset[str] = frozenset({
    "ele", "ela", "eles", "elas", "isso", "aquilo", "este", "esta",
    "estes", "estas", "aquele", "aquela", "aqueles", "aquelas",
    "empresa", "grupo", "homem", "mulher", "pessoa", "indivíduo",
    "organização", "entidade", "outro", "outra", "outros", "outras",
    "n/a", "unknown", "desconhecido", "desconhecida", "nenhum", "nenhuma",
    "algo", "alguém", "ninguém",
})

_DATE_RE = re.compile(
    r"^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$"
    r"|^\d{4}$"
    r"|^\d{1,2}\s+de\s+\w+\s+de\s+\d{4}$",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"^\$?\d[\d,\.]*%?$")


def _normalize_entity(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x1f\x7f​‌‍﻿]", " ", text)
    text = " ".join(text.split())
    if text == text.lower() or text == text.upper():
        text = text.title()
    return text.strip()


def _resolve_entity(entity: str) -> str:
    canonical = ENTITY_ALIASES.get(entity)
    if canonical:
        return canonical
    for alias, canon in ENTITY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", entity, re.IGNORECASE):
            if entity.lower() != canon.lower():
                return canon
    return entity


def _normalize_predicate(predicate: str) -> str | None:
    p = predicate.strip().upper()
    if p in PREDICATE_ALIASES:
        p = PREDICATE_ALIASES[p]
    return p if p in VALID_PREDICATES else None


def _is_valid(subject: str, obj: str, confidence: float) -> bool:
    if confidence < MIN_CONFIDENCE:
        return False
    for entity in (subject, obj):
        if len(entity) < MIN_ENTITY_LENGTH:
            return False
        if entity.lower() in ENTITY_BLOCKLIST:
            return False
        if _DATE_RE.match(entity):
            return False
        if _NUMBER_RE.match(entity):
            return False
    return subject != obj


def build_graph(source_dir: Path = SAIDA_DIR) -> nx.MultiDiGraph:
    """
    Lê os JSONs de source_dir e constrói um MultiDiGraph com metadados.

    Nós:  entity_type, aliases, degree
    Arestas: relation, confidence, source_file, weight
    """
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    node_aliases: dict[str, set[str]] = {}
    seen_edges: set[tuple[str, str, str]] = set()

    for path in sorted(source_dir.glob("*.json")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            raw_list = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw_list, list):
            continue

        for raw in raw_list:
            if not isinstance(raw, dict):
                continue

            subject_raw = str(raw.get("subject", "")).strip()
            obj_raw     = str(raw.get("object", "")).strip()
            predicate   = str(raw.get("predicate", "")).strip()
            confidence  = raw.get("confidence", 0.0)
            subject_type = str(raw.get("subject_type") or "Unknown")
            object_type  = str(raw.get("object_type")  or "Unknown")

            if not isinstance(confidence, (int, float)):
                continue
            confidence = float(confidence)

            subject = _resolve_entity(_normalize_entity(subject_raw))
            obj     = _resolve_entity(_normalize_entity(obj_raw))
            predicate = _normalize_predicate(predicate) or ""

            if not predicate or not _is_valid(subject, obj, confidence):
                continue

            # Deduplicação de arestas
            edge_key = (subject, predicate, obj)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Nós
            for node, etype, orig in (
                (subject, subject_type, subject_raw),
                (obj,     object_type,  obj_raw),
            ):
                if not G.has_node(node):
                    G.add_node(node, entity_type=etype, aliases=[])
                    node_aliases[node] = set()
                if G.nodes[node].get("entity_type") in ("Unknown", None) and etype not in ("Unknown", None):
                    G.nodes[node]["entity_type"] = etype
                normalized_orig = _normalize_entity(orig)
                if normalized_orig != node:
                    node_aliases[node].add(normalized_orig)

            G.add_edge(
                subject,
                obj,
                relation=predicate,
                confidence=confidence,
                source_file=path.name,
                weight=1.0,
            )

    # Finaliza metadados dos nós
    for node in G.nodes():
        G.nodes[node]["aliases"] = sorted(node_aliases.get(node, set()))
        G.nodes[node]["degree"]  = G.degree(node)

    return G
