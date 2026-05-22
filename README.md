# python-graph

Pipeline de extração de conhecimento e construção de grafos semânticos a partir de documentos de texto, usando LLMs (Gemini ou GPT-4o) para identificar entidades e relações.

---

## Visão geral

O pipeline opera em duas etapas independentes:

```
Documentos .md
      │
      ▼
  [main.py]  ──── LLM (Gemini ou GPT-4o) ────▶  JSON de relações (saida_json/)
      │
      ▼
 [graph.py]  ──── pipeline de 10 etapas ──────▶  Grafo (grafos/)
```

1. **`main.py`** — lê arquivos `.md`, envia o conteúdo a uma LLM e extrai relações estruturadas no formato `(sujeito, predicado, objeto)`, salvando como JSON.
2. **`graph.py`** — consome os JSONs, valida, normaliza, deduplica e constrói um grafo de conhecimento exportado em múltiplos formatos.

---

## Estrutura do repositório

```
.
├── main.py           # Extração de relações via LLM
├── graph.py          # Construção e exportação do grafo
├── pyproject.toml    # Dependências do projeto
├── .env              # Chaves de API (não commitado)
├── .env.example      # Modelo de variáveis de ambiente
├── saida_json/       # JSONs brutos gerados pelo main.py
├── saida/            # JSONs de entrada para o graph.py
├── grafos/           # Saídas do grafo (GEXF, GraphML, JSON, HTML)
└── logs/
    ├── processados.json        # Controle de arquivos já processados
    ├── pipeline.log            # Log detalhado do pipeline
    ├── pipeline_metrics.json   # Métricas da última execução
    └── pipeline_discarded.json # Relações descartadas (até 500 amostras)
```

---

## Pré-requisitos

- Python 3.13+
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

Instale também as dependências das APIs:

```bash
uv add google-generativeai openai python-dotenv
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
```

Onde obter as chaves:
- **Gemini**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

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

## Modelos utilizados

| Provider | Modelo | Flag |
|---|---|---|
| Google Gemini | `gemini-2.0-flash` | `--provider gemini` (padrão) |
| OpenAI | `gpt-4o` | `--provider openai` |

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
