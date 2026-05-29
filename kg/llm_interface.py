"""
Interface conversacional LLM ↔ Knowledge Graph via function calling.

Fluxo:
    Usuário → LLM (interpreta + seleciona ferramenta)
            → KGQuery (executa no grafo)
            → LLM (formata resposta em linguagem natural)
            → Resposta final ao usuário

Suporta: OpenAI (gpt-4o), Gemini (gemini-2.0-flash) e Anthropic (claude-opus-4-7)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from dotenv import load_dotenv

from .queries import EntityInfo, KGQuery, PathInfo, RelationInfo

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Você é um analista especializado no Knowledge Graph do caso Jeffrey Epstein.

Você tem acesso a um grafo de conhecimento construído a partir de documentos
oficiais, com entidades (pessoas, organizações, locais, empresas, etc.) e
relações entre elas (ASSOCIATED_WITH, WORKS_FOR, FINANCED, TRANSFERRED_MONEY_TO, etc.).

SUAS RESPONSABILIDADES:
- Responder perguntas sobre entidades, conexões e estrutura do grafo
- Usar as ferramentas disponíveis para buscar informações precisas
- Interpretar métricas de rede (PageRank, betweenness, etc.) de forma acessível
- Apresentar caminhos relacionais de forma clara e compreensível
- Sempre fundamentar a resposta nos dados retornados pelas ferramentas

MÉTRICAS — COMO INTERPRETAR:
- PageRank alto     → entidade influente de forma indireta (muitas conexões chegam até ela)
- Betweenness alto  → broker relacional: intermediário entre grupos distintos
- Degree alto       → entidade com muitas conexões diretas (hub)
- Closeness alto    → entidade estruturalmente próxima do centro do grafo

INSTRUÇÕES OBRIGATÓRIAS:
- Use SEMPRE pelo menos uma ferramenta antes de responder
- Se a entidade não for encontrada, chame search_entity primeiro
- Para "quem é mais importante/influente/central" → use get_central_nodes
- Para "como A se conecta a B" ou "ligação entre A e B" → use find_path
- Para "o que A faz / quem é A" → use get_entity_info ou get_llm_context
- Para "quantos nós/arestas existem" → use get_graph_stats
- Para "liste todas as organizações/pessoas" → use find_by_type
- Responda em Português, de forma clara, objetiva e analítica
- Ao citar métricas numéricas, explique o que significam em contexto\
"""

# ─────────────────────────────────────────────────────────────────────────────
# DEFINIÇÃO DAS FERRAMENTAS (formato OpenAI — fonte canônica)
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS_OPENAI: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_entity",
            "description": (
                "Busca entidades no grafo cujo nome contenha o termo pesquisado (case-insensitive). "
                "Sempre use antes de outras ferramentas quando não souber o nome exato."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome ou parte do nome da entidade a buscar",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_info",
            "description": (
                "Retorna informações completas de uma entidade: tipo, grau, "
                "degree_centrality, betweenness_centrality, closeness_centrality, "
                "eigenvector_centrality, PageRank, community_id e vizinhos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome exato (ou parcial) da entidade no grafo",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_graph_stats",
            "description": (
                "Retorna estatísticas globais do grafo: total de nós, arestas, "
                "comunidades detectadas, tipos de relação e densidade."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_central_nodes",
            "description": (
                "Retorna as entidades mais centrais do grafo segundo uma métrica. "
                "Use para responder 'quem é mais importante/influente/estratégico'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": [
                            "pagerank",
                            "degree_centrality",
                            "betweenness_centrality",
                            "closeness_centrality",
                            "eigenvector_centrality",
                        ],
                        "description": "Métrica de centralidade a utilizar",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Quantidade de entidades a retornar (padrão: 10)",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": (
                            "Filtrar por tipo: 'Pessoa', 'Organização', 'Empresa', "
                            "'País', 'Cidade', 'Local', 'Instituição', 'Evento', "
                            "'Documento', 'Operação Financeira'. Omitir para todos os tipos."
                        ),
                    },
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_path",
            "description": (
                "Encontra o caminho mais curto entre duas entidades no grafo (direcionado). "
                "Use para responder 'como A se conecta a B' ou 'qual a ligação entre A e B'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Nome da entidade de origem",
                    },
                    "target": {
                        "type": "string",
                        "description": "Nome da entidade de destino",
                    },
                },
                "required": ["source", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": (
                "Retorna predecessores e sucessores diretos de uma entidade. "
                "Use para explorar o entorno imediato de um nó."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Nome da entidade",
                    }
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_community",
            "description": (
                "Retorna todos os membros da comunidade/cluster ao qual a entidade pertence. "
                "Comunidades são grupos densamente conectados detectados pelo algoritmo Louvain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Nome da entidade",
                    }
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_relations",
            "description": (
                "Retorna todas as relações de uma entidade com seus predicados e confiança. "
                "Use para listar conexões com tipo (WORKS_FOR, FINANCED, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Nome da entidade",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["out", "in", "both"],
                        "description": "Direção: 'out' (saída), 'in' (entrada), 'both' (ambas, padrão)",
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_relations_between",
            "description": (
                "Retorna todas as relações diretas entre duas entidades específicas (qualquer direção). "
                "Use quando quer saber exatamente o que conecta A a B diretamente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_a": {
                        "type": "string",
                        "description": "Nome da primeira entidade",
                    },
                    "entity_b": {
                        "type": "string",
                        "description": "Nome da segunda entidade",
                    },
                },
                "required": ["entity_a", "entity_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_by_type",
            "description": (
                "Lista todas as entidades de um tipo específico, ordenadas por PageRank. "
                "Use para responder 'liste as organizações' ou 'quais são as pessoas do grafo'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "description": (
                            "Tipo da entidade: 'Pessoa', 'Organização', 'Empresa', 'País', "
                            "'Cidade', 'Local', 'Instituição', 'Evento', 'Documento', "
                            "'Operação Financeira'"
                        ),
                    }
                },
                "required": ["entity_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_llm_context",
            "description": (
                "Retorna contexto estrutural enriquecido sobre uma entidade: relações diretas, "
                "comunidade, nós próximos (ego_graph) e entidades centrais da mesma comunidade. "
                "Use quando precisar de visão abrangente para responder perguntas complexas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Nome da entidade",
                    }
                },
                "required": ["entity"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def _serialize(result: Any) -> str:
    """Converte qualquer resultado de KGQuery para string JSON."""
    if result is None:
        return json.dumps({"result": None}, ensure_ascii=False)
    if isinstance(result, (EntityInfo, RelationInfo, PathInfo)):
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)
    if isinstance(result, list):
        items = [
            asdict(item) if isinstance(item, (EntityInfo, RelationInfo, PathInfo)) else item
            for item in result
        ]
        return json.dumps(items, ensure_ascii=False, indent=2)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return json.dumps({"result": str(result)}, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI TOOL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_anthropic_tools() -> list[dict]:
    """Converte _TOOLS_OPENAI para formato Anthropic (input_schema)."""
    result = []
    for tool in _TOOLS_OPENAI:
        fn = tool["function"]
        result.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return result


_TOOLS_ANTHROPIC: list[dict] = _build_anthropic_tools()


def _build_gemini_tool():
    """Converte _TOOLS_OPENAI para o formato google.genai.types.Tool."""
    from google.genai import types

    _TYPE_MAP = {
        "string":  types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number":  types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
        "object":  types.Type.OBJECT,
        "array":   types.Type.ARRAY,
    }

    declarations = []
    for tool in _TOOLS_OPENAI:
        fn = tool["function"]
        props = fn["parameters"].get("properties", {})
        required = fn["parameters"].get("required", [])

        gemini_props: dict[str, types.Schema] = {}
        for param_name, param_schema in props.items():
            gemini_props[param_name] = types.Schema(
                type=_TYPE_MAP.get(param_schema.get("type", "string"), types.Type.STRING),
                description=param_schema.get("description", ""),
                enum=param_schema.get("enum"),
            )

        declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn["description"],
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties=gemini_props,
                required=required,
            ) if gemini_props else None,
        ))

    return types.Tool(functionDeclarations=declarations)


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class KGInterface:
    """
    Interface conversacional LLM ↔ Knowledge Graph.

    Uso:
        from kg import KGRuntime
        from kg.llm_interface import KGInterface

        rt    = KGRuntime()
        iface = KGInterface(rt.query, provider="anthropic")
        print(iface.chat("Qual é o nó mais central do grafo?"))
        print(iface.chat("Como Jeffrey Epstein se conecta a Ghislaine Maxwell?"))
        iface.reset()  # limpa histórico de conversa
    """

    _MAX_TOOL_ROUNDS = 8

    def __init__(
        self,
        query: KGQuery,
        provider: str = "anthropic",
        model: str | None = None,
        verbose: bool = False,
    ) -> None:
        self._q = query
        self._provider = provider.lower()
        self._verbose = verbose
        # Histórico: lista de {"role": "user"|"assistant", "content": str}
        self._history: list[dict[str, str]] = []

        if self._provider == "openai":
            from openai import OpenAI
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._model = model or "gpt-4o"

        elif self._provider == "anthropic":
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self._model = model or "claude-opus-4-7"

        elif self._provider == "openrouter":
            from openai import OpenAI
            self._openai = OpenAI(
                api_key=os.getenv("OPEN_ROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
            self._model = model or "google/gemma-4-31b-it:free"

        elif self._provider == "gemini":
            import google.genai as genai
            from google.genai import types as gtypes
            self._genai = genai
            self._gtypes = gtypes
            self._gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self._model = model or "gemini-2.0-flash"
            self._gemini_tool = _build_gemini_tool()

        else:
            raise ValueError(
                f"Provider '{provider}' não suportado. Use 'openai', 'anthropic', 'openrouter' ou 'gemini'."
            )

    # ──────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """Envia mensagem e retorna resposta em linguagem natural."""
        if self._provider in ("openai", "openrouter"):
            return self._openai_loop(user_input)
        if self._provider == "anthropic":
            return self._claude_loop(user_input)
        return self._gemini_loop(user_input)

    def reset(self) -> None:
        """Limpa o histórico de conversa."""
        self._history.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        """Retorna cópia do histórico de conversa (user/assistant)."""
        return list(self._history)

    # ──────────────────────────────────────────────────────────────────────
    # Dispatch de ferramentas
    # ──────────────────────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> str:
        q = self._q
        dispatch = {
            "search_entity":       lambda: q.search_entity(args["name"]),
            "get_entity_info":     lambda: q.get_entity(args["name"]),
            "get_graph_stats":     lambda: q.get_graph_stats(),
            "get_central_nodes":   lambda: q.central_nodes(
                metric=args.get("metric", "pagerank"),
                n=int(args.get("top_n", 10)),
                entity_type=args.get("entity_type"),
            ),
            "find_path":           lambda: q.shortest_path(args["source"], args["target"]),
            "get_neighbors":       lambda: q.neighbors(args["entity"]),
            "get_community":       lambda: q.get_community(args["entity"]),
            "get_relations":       lambda: q.get_relations(
                args["entity"], direction=args.get("direction", "both")
            ),
            "get_relations_between": lambda: q.get_relations_between(
                args["entity_a"], args["entity_b"]
            ),
            "find_by_type":        lambda: q.find_by_type(args["entity_type"]),
            "get_llm_context":     lambda: q.llm_context(args["entity"]),
        }

        fn = dispatch.get(name)
        if fn is None:
            result = {"error": f"Ferramenta desconhecida: {name}"}
        else:
            try:
                result = fn()
            except Exception as exc:
                result = {"error": str(exc)}

        serialized = _serialize(result)
        if self._verbose:
            preview = serialized[:150].replace("\n", " ")
            print(f"  [tool] {name}({json.dumps(args, ensure_ascii=False)}) → {preview}…")
        return serialized

    # ──────────────────────────────────────────────────────────────────────
    # OpenAI loop
    # ──────────────────────────────────────────────────────────────────────

    def _openai_loop(self, user_input: str) -> str:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        for turn in self._history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": user_input})

        for _ in range(self._MAX_TOOL_ROUNDS):
            response = self._openai.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=_TOOLS_OPENAI,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                answer = msg.content or ""
                self._history.append({"role": "user",      "content": user_input})
                self._history.append({"role": "assistant", "content": answer})
                return answer

            # Adiciona mensagem do assistant com tool_calls
            messages.append(msg.model_dump(exclude_unset=True, exclude_none=True))

            # Executa cada tool call e adiciona resultado
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                tool_result = self._execute_tool(tc.function.name, args)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      tool_result,
                })

        return "Não foi possível concluir a consulta após múltiplas tentativas."

    # ──────────────────────────────────────────────────────────────────────
    # Anthropic (Claude) loop
    # ──────────────────────────────────────────────────────────────────────

    def _claude_loop(self, user_input: str) -> str:
        messages: list[dict] = []

        for turn in self._history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": user_input})

        for _ in range(self._MAX_TOOL_ROUNDS):
            response = self._anthropic.messages.create(
                model=self._model,
                max_tokens=16000,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS_ANTHROPIC,
                messages=messages,
            )

            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks  = [b for b in response.content if b.type == "text"]

            if not tool_blocks:
                answer = text_blocks[0].text if text_blocks else ""
                self._history.append({"role": "user",      "content": user_input})
                self._history.append({"role": "assistant", "content": answer})
                return answer

            # Adiciona turno do assistente (pode conter texto + tool_use)
            messages.append({"role": "assistant", "content": response.content})

            # Executa ferramentas e empacota resultados num único turno de usuário
            tool_results = []
            for tb in tool_blocks:
                tool_result = self._execute_tool(tb.name, tb.input)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tb.id,
                    "content":     tool_result,
                })

            messages.append({"role": "user", "content": tool_results})

        return "Não foi possível concluir a consulta após múltiplas tentativas."

    # ──────────────────────────────────────────────────────────────────────
    # Gemini loop
    # ──────────────────────────────────────────────────────────────────────

    def _gemini_loop(self, user_input: str) -> str:
        types = self._gtypes

        # Reconstrói histórico como lista de Content
        contents: list = []
        for turn in self._history:
            role = "user" if turn["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=turn["content"])],
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        ))

        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            tools=[self._gemini_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
        )

        for _ in range(self._MAX_TOOL_ROUNDS):
            response = self._gemini_client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            candidate = response.candidates[0]
            model_content = candidate.content

            # Coleta chamadas de função da resposta
            func_calls = [
                p.function_call
                for p in model_content.parts
                if p.function_call and p.function_call.name
            ]

            if not func_calls:
                answer = response.text or ""
                self._history.append({"role": "user",      "content": user_input})
                self._history.append({"role": "assistant", "content": answer})
                return answer

            # Adiciona a resposta do modelo (com function_calls) ao histórico
            contents.append(model_content)

            # Executa todas as ferramentas e prepara respostas em um único turno
            response_parts = []
            for fc in func_calls:
                tool_result_str = self._execute_tool(fc.name, dict(fc.args))
                tool_result_data = json.loads(tool_result_str)
                response_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": tool_result_data},
                    )
                ))

            contents.append(types.Content(role="user", parts=response_parts))

        return "Não foi possível concluir a consulta após múltiplas tentativas."
