# python-graph

Pipeline de extração de conhecimento, construção e análise de grafos semânticos a partir de documentos de texto, com interface conversacional LLM via function calling.

---

## Visão geral

O sistema opera em três camadas independentes:

```
Documentos .md
      │
      ▼
  [main.py]  ──── LLM (Gemini ou GPT-4o) ────▶  JSON de relações (saida_json/)
      │
      ▼
 [graph.py]  ──── pipeline de 10 etapas ──────▶  Grafo base (grafos/)
      │
      ▼
   [kg/]     ──── NetworkX analítico ──────────▶  Grafo analítico + métricas + comunidades
      │
      ▼
[llm_interface.py] ── function calling ────────▶  Interface conversacional em linguagem natural
      │
      ▼
  [chat.py]  ──── loop interativo ─────────────▶  Terminal (perguntas em linguagem natural)
```

1. **`main.py`** — lê arquivos `.md`, envia o conteúdo a uma LLM e extrai relações estruturadas no formato `(sujeito, predicado, objeto)`, salvando como JSON.
2. **`graph.py`** — consome os JSONs, valida, normaliza, deduplica e constrói um grafo de conhecimento exportado em múltiplos formatos.
3. **`kg/`** — camada analítica sobre o grafo: métricas de centralidade, detecção de comunidades, consultas relacionais e interface de conversação com LLM via function calling.

---

## Estrutura do repositório

```
.
├── main.py               # Extração de relações via LLM
├── graph.py              # Construção e exportação do grafo base
├── chat.py               # Loop interativo de consulta via LLM
├── pyproject.toml        # Dependências do projeto
├── .env                  # Chaves de API (não commitado)
├── .env.example          # Modelo de variáveis de ambiente
│
├── kg/                   # Camada analítica NetworkX
│   ├── __init__.py       # Exporta KGRuntime e KGInterface
│   ├── graph_builder.py  # Constrói MultiDiGraph dos JSONs tratados
│   ├── metrics.py        # Centralidades (degree, betweenness, closeness, eigenvector, pagerank)
│   ├── communities.py    # Detecção de comunidades via Louvain
│   ├── queries.py        # KGQuery — interface de consulta relacional
│   ├── exporters.py      # Exportação analítica (GEXF, GraphML, Pickle)
│   ├── graph_runtime.py  # KGRuntime — orquestrador do pipeline analítico
│   └── llm_interface.py  # KGInterface — interface LLM via function calling
│
├── saida_json/           # JSONs brutos gerados pelo main.py
├── saida/                # JSONs normalizados (entrada para kg/)
├── grafos/               # Saídas exportadas
│   ├── graph.*           # Grafo base (graph.py): GEXF, GraphML, JSON, HTML
│   └── kg_analytic.*     # Grafo analítico (kg/): GEXF, GraphML, Pickle
└── logs/
    ├── processados.json        # Controle de arquivos já processados
    ├── pipeline.log            # Log detalhado do pipeline
    ├── pipeline_metrics.json   # Métricas da última execução
    └── pipeline_discarded.json # Relações descartadas (até 500 amostras)
```

---

## Pré-requisitos

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (gerenciador de pacotes)

Instale as dependências:

```bash
uv sync
```

As dependências declaradas em `pyproject.toml` são:

| Pacote | Uso |
|---|---|
| `networkx` | Estrutura e algoritmos de grafo |
| `matplotlib` | Paleta de cores dos nós |
| `pydantic` | Validação de esquema das relações |
| `pyvis` | Exportação HTML interativa |
| `python-louvain` | Detecção de comunidades (Louvain) |
| `scipy` | Dependência de algoritmos de centralidade |

Instale também as dependências das APIs:

```bash
uv add google-genai openai python-dotenv
```

---

## Configuração

Crie um arquivo `.env` na raiz do projeto (use `.env.example` como base):

```bash
cp .env.example .env
```

Preencha com suas chaves:

```env
GEMINI_API_KEY="sua-chave-aqui"
GPT_API_KEY="sua-chave-aqui"
ANTHROPIC_API_KEY="sua-chave-aqui"
OPEN_ROUTER_API_KEY="sua-chave-aqui"
```

Onde obter as chaves:
- **Gemini**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- **OpenRouter**: [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)

> O arquivo `.env` está no `.gitignore` e nunca será commitado.

---

## Uso

### Etapa 1 — Extração de relações (`main.py`)

Processa todos os arquivos `.md` do diretório de entrada e gera JSONs de relações.

```bash
# Usando Gemini (padrão)
python main.py

# Usando GPT-4o
python main.py --provider openai
```

O script mantém um log de arquivos já processados em `logs/processados.json`. Arquivos já processados são ignorados em execuções subsequentes (`[SKIP]`).

**Saída:** arquivos `.json` em `saida_json/`, um por arquivo `.md`.

---

### Etapa 2 — Construção do grafo (`graph.py`)

Lê os JSONs de relações, executa o pipeline de 10 etapas e exporta o grafo.

```bash
python graph.py
```

**Saída em `grafos/`:**

| Arquivo | Formato | Uso recomendado |
|---|---|---|
| `graph.gexf` | GEXF | Gephi, análise avançada |
| `graph.graphml` | GraphML | yEd, ferramentas XML |
| `graph.json` | JSON | Integrações customizadas |
| `graph.html` | HTML interativo | Visualização no browser |

---

## Pipeline de 10 etapas (`graph.py`)

| Etapa | Nome | Descrição |
|---|---|---|
| 1 | Leitura | Carrega arquivos `.json` da pasta `saida/` |
| 2 | Validação | Verifica o esquema de cada relação via Pydantic |
| 3 | Normalização | Corrige Unicode, espaços e capitalização das entidades |
| 4 | Aliases de entidades | Unifica variações (`"Epstein"` → `"Jeffrey Epstein"`) |
| 5 | Aliases de predicados | Mapeia predicados equivalentes para o padrão |
| 6 | Filtro de qualidade | Remove relações com confidence < 0.70, entidades bloqueadas, self-loops |
| 7 | Deduplicação | Remove relações com a mesma tripla `(sujeito, predicado, objeto)` |
| 8 | Construção do grafo | Cria um `MultiDiGraph` com NetworkX |
| 9 | Exportação | Gera GEXF, GraphML, JSON e HTML |
| 10 | Métricas | Salva relatório e amostra de relações descartadas |

---

## Esquema das relações

Cada relação extraída segue o formato:

```json
{
  "subject": "Jeffrey Epstein",
  "predicate": "ASSOCIATED_WITH",
  "object": "Ghislaine Maxwell",
  "subject_type": "Pessoa",
  "object_type": "Pessoa",
  "confidence": 0.97
}
```

### Predicados válidos

```
ASSOCIATED_WITH    WORKS_FOR          FOUNDED            OWNS
LOCATED_IN         PART_OF            INVESTIGATED       ACCUSED_OF
PARTICIPATED_IN    COMMUNICATED_WITH  TRANSFERRED_MONEY_TO  RESPONSIBLE_FOR
RELATED_TO         MEMBER_OF          MET_WITH           TRAVELED_TO
FINANCED           MANAGES            REPRESENTS         CONNECTED_TO
```

### Tipos de entidades

`Pessoa` · `Organização` · `Empresa` · `País` · `Cidade` · `Local` · `Instituição` · `Evento` · `Documento` · `Operação Financeira`

---

## Camada Analítica (kg/)

A camada `kg/` constrói um grafo analítico enriquecido sobre os mesmos JSONs do `graph.py`, adicionando métricas de rede, detecção de comunidades e uma interface de consulta programática.

### Uso rápido

```python
from kg import KGRuntime

rt = KGRuntime()          # constrói grafo, calcula métricas, detecta comunidades, exporta
rt.print_summary()        # resumo: nós, arestas, comunidades, top hubs

q = rt.query              # interface de consulta (KGQuery)

# Busca de entidades
q.search_entity("Epstein")                    # busca fuzzy por nome/alias
q.get_entity("Jeffrey Epstein")              # metadados completos + métricas

# Relações
q.get_relations("Jeffrey Epstein", direction="out")
q.get_relations_between("Jeffrey Epstein", "Ghislaine Maxwell")

# Caminhos
q.shortest_path("Jeffrey Epstein", "Ghislaine Maxwell")
q.all_simple_paths("FBI", "Ghislaine Maxwell", cutoff=5)

# Vizinhança
q.neighbors("Jeffrey Epstein")
q.ego_graph("Jeffrey Epstein", radius=2)
q.descendants("Jeffrey Epstein")

# Comunidades
q.get_community("Jeffrey Epstein")
q.central_nodes(metric="pagerank", n=10)
q.central_nodes(metric="betweenness_centrality", entity_type="Pessoa")

# Estatísticas e tipos
q.get_graph_stats()
q.find_by_type("Organização")
q.get_all_relation_types()

# Contexto para LLM (GraphRAG)
q.llm_context("Jeffrey Epstein")
```

O `KGRuntime` usa cache via Pickle: na primeira execução constrói e exporta o grafo; nas seguintes carrega em milissegundos.

---

## Interface LLM + GraphRAG (KGInterface)

`KGInterface` é uma interface conversacional que conecta perguntas em linguagem natural ao Knowledge Graph via **function calling**. A LLM recebe a pergunta, seleciona e executa ferramentas do grafo, e formata a resposta de forma analítica.

### Fluxo

```
Usuário → LLM (interpreta + seleciona ferramenta)
        → KGQuery (executa no grafo NetworkX)
        → LLM (formata resposta em linguagem natural)
        → Resposta contextualizada
```

### Uso

```python
from kg import KGRuntime, KGInterface

rt    = KGRuntime()
iface = KGInterface(rt.query, provider="openai")   # ou provider="gemini"

print(iface.chat("Qual é o nó mais central do grafo?"))
print(iface.chat("Como Jeffrey Epstein se conecta a Ghislaine Maxwell?"))
print(iface.chat("Liste as organizações mais influentes."))
print(iface.chat("Quem são os intermediários estratégicos do grafo?"))

iface.reset()   # limpa histórico de conversa
```

### Ferramentas disponíveis para a LLM

| Ferramenta | Descrição |
|---|---|
| `search_entity` | Busca fuzzy de entidades por nome |
| `get_entity_info` | Metadados completos + métricas de uma entidade |
| `get_graph_stats` | Estatísticas globais (nós, arestas, comunidades, densidade) |
| `get_central_nodes` | Top entidades por métrica (pagerank, betweenness, etc.) |
| `find_path` | Caminho mais curto entre duas entidades |
| `get_neighbors` | Predecessores e sucessores diretos |
| `get_community` | Membros do cluster/comunidade de uma entidade |
| `get_relations` | Todas as relações de uma entidade (entrada/saída) |
| `get_relations_between` | Relações diretas entre dois nós específicos |
| `find_by_type` | Lista entidades por tipo (Pessoa, Organização, etc.) |
| `get_llm_context` | Contexto estrutural enriquecido (relações + comunidade + ego_graph) |

### Métricas — como a LLM as interpreta

| Métrica | Significado |
|---|---|
| PageRank alto | Entidade influente indiretamente (muitas referências chegam até ela) |
| Betweenness alto | Broker relacional: intermediário entre grupos distintos |
| Degree alto | Hub: entidade com muitas conexões diretas |
| Closeness alto | Entidade estruturalmente próxima do centro do grafo |

---

## Chat interativo (`chat.py`)

`chat.py` é o ponto de entrada para consultas em linguagem natural ao Knowledge Graph via terminal. Carrega o grafo, inicializa a interface LLM e mantém um loop de perguntas e respostas com histórico de conversa.

### Uso

```bash
# Provider padrão (OpenRouter, modelo gratuito)
python chat.py

# Outros providers
python chat.py --provider anthropic
python chat.py --provider openai
python chat.py --provider gemini

# Modelo específico
python chat.py --provider openrouter --model "meta-llama/llama-3.3-70b-instruct:free"

# Exibir chamadas de ferramenta e dados retornados
python chat.py --verbose
```

### Comandos durante a sessão

| Comando | Ação |
|---|---|
| `/stats` | Exibe estatísticas do grafo (nós, arestas, comunidades, densidade) |
| `/reset` | Limpa o histórico de conversa |
| `/modelo` | Exibe o provider e modelo em uso |
| `sair` | Encerra a sessão |

### Providers e modelos padrão

| Provider | Modelo padrão |
|---|---|
| `openrouter` | `google/gemma-4-31b-it:free` |
| `openai` | `gpt-4o` |
| `anthropic` | `claude-opus-4-7` |
| `gemini` | `gemini-2.0-flash` |

> **OpenRouter** oferece acesso gratuito a diversos modelos. Modelos com suporte a function calling e sufixo `:free` funcionam com este sistema. Caso um modelo retorne erro 429 (rate limit), troque pelo flag `--model`.

---

## Modelos utilizados

| Provider | Modelo | Flag / parâmetro |
|---|---|---|
| Google Gemini | `gemini-2.0-flash` | `--provider gemini` (padrão em main.py) |
| OpenAI | `gpt-4o` | `--provider openai` |
| KGInterface | configurável | `KGInterface(query, provider="openai", model="gpt-4o")` |

---

## Logs e diagnóstico

Após executar `graph.py`, os logs ficam em `logs/`:

```bash
# Log em tempo real
tail -f logs/pipeline.log

# Métricas da última execução
cat logs/pipeline_metrics.json

# Relações descartadas (para diagnóstico)
cat logs/pipeline_discarded.json
```

Exemplo de métricas:

```json
{
  "files": { "processed": 42, "invalid": 0 },
  "relations": {
    "loaded": 1840,
    "discarded": {
      "low_confidence": 123,
      "blocked_entity": 45,
      "duplicates": 210
    },
    "final": 1462
  },
  "graph": { "nodes": 318, "edges": 1462 }
}
```
