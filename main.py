import os
import re
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

# =========================
# CONFIG
# =========================

INPUT_DIR = "/home/luca/Documentos/rc/epstein-data/mdf"
OUTPUT_DIR = "/home/luca/Documentos/rc/convert-data/saida_json"
LOG_FILE = "/home/luca/Documentos/rc/convert-data/logs/processados.json"

SYSTEM_PROMPT = """
Você é um sistema especializado em extração de conhecimento estruturado para construção de grafos semânticos.

Sua tarefa é analisar textos e extrair APENAS relações factuais, explícitas e semanticamente relevantes.

OBJETIVO:
Extrair entidades e relações úteis para construção de um grafo de conhecimento consistente e consultável.

REGRAS GERAIS:

- Utilize APENAS entidades nomeadas explícitas presentes no texto.
- Nunca utilize pronomes como:
  "ele", "ela", "eles", "isso", "aquilo", etc.
- Normalize entidades quando possível.
- Preserve nomes próprios completos.
- Todas as entidades e relações devem estar em PORTUGUÊS.
- Extraia apenas relações factuais e relevantes.
- Ignore conteúdo especulativo, ambíguo ou opinativo.
- Ignore relações fracas ou genéricas.

NÃO EXTRAIA:

- Datas isoladas
- Horários
- Números sem contexto
- Relações vagas
- Frases descritivas sem interação entre entidades
- Verbos genéricos como:
  "disse", "comentou", "falou", "mencionou"
- Relações sem valor semântico para grafos
- Entidades genéricas como:
  "empresa", "homem", "mulher", "grupo"
- Informações duplicadas

TIPOS DE ENTIDADES PERMITIDOS:

- Pessoa
- Organização
- Empresa
- País
- Cidade
- Local
- Instituição
- Evento
- Documento
- Operação Financeira

UTILIZE APENAS OS PREDICADOS ABAIXO:

- ASSOCIATED_WITH
- WORKS_FOR
- FOUNDED
- OWNS
- LOCATED_IN
- PART_OF
- INVESTIGATED
- ACCUSED_OF
- PARTICIPATED_IN
- COMMUNICATED_WITH
- TRANSFERRED_MONEY_TO
- RESPONSIBLE_FOR
- RELATED_TO
- MEMBER_OF
- MET_WITH
- TRAVELED_TO
- FINANCED
- MANAGES
- REPRESENTS
- CONNECTED_TO

REGRAS DOS PREDICADOS:

- Escolha SEMPRE o predicado mais específico possível.
- Nunca invente novos predicados.
- Nunca utilize linguagem natural livre como predicado.
- Todos os predicados devem permanecer exatamente como definidos acima.

FORMATO OBRIGATÓRIO DE SAÍDA:

[
  {
    "subject": "Entidade origem",
    "predicate": "PREDICADO_PADRONIZADO",
    "object": "Entidade destino",
    "subject_type": "Tipo da entidade origem",
    "object_type": "Tipo da entidade destino",
    "confidence": 0.95
  }
]

REGRAS DE QUALIDADE:

- Confidence deve variar entre 0.0 e 1.0
- Utilize confidence alta apenas quando a relação for explícita.
- Não gere relações inferidas.
- Não gere relações implícitas.
- Não gere relações sem evidência textual clara.
- Evite duplicações semânticas.

EXEMPLO VÁLIDO:

[
  {
    "subject": "Jeffrey Epstein",
    "predicate": "ASSOCIATED_WITH",
    "object": "Ghislaine Maxwell",
    "subject_type": "Pessoa",
    "object_type": "Pessoa",
    "confidence": 0.97
  }
]

EXEMPLO INVÁLIDO:

[
  ["EUA", "na data", "08/12"]
]

MOTIVOS:
- predicado inválido
- data isolada
- relação sem valor semântico

Retorne APENAS JSON válido.
"""


PROMPT_TEMPLATE = """
Extraia relações de conhecimento do texto abaixo seguindo rigorosamente o schema e as regras definidas.

TEXTO:
{content}

Retorne APENAS um array JSON válido.
"""

# =========================
# CLIENTS
# =========================

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-3.5-flash")

openai_client = OpenAI(api_key=os.getenv("GPT_API_KEY"))

def gerar_resposta(prompt, provider):
    if provider == "openai":
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # gemini (default)
    response = gemini_model.generate_content(
        [
            {"role": "user", "parts": [SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Entendido. Aguardando texto para extração."]},
            {"role": "user", "parts": [prompt]},
        ]
    )
    return response.text.strip()

# =========================
# LOG
# =========================

def carregar_processados():

    if not os.path.exists(LOG_FILE):
        return {}

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def salvar_processados(data):

    os.makedirs("logs", exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# =========================
# PROCESSAMENTO
# =========================

def processar_arquivo(path_arquivo, provider):

    with open(path_arquivo, "r", encoding="utf-8") as f:
        conteudo = f.read()

    prompt = PROMPT_TEMPLATE.format(content=conteudo)

    texto = gerar_resposta(prompt, provider)
    texto = re.sub(r"^```(?:json)?\s*", "", texto)
    texto = re.sub(r"\s*```$", "", texto)

    try:
        dados = json.loads(texto)
        if not isinstance(dados, list):
            raise ValueError("resposta não é uma lista JSON")
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"JSON inválido: {e}\nResposta bruta: {texto[:300]}") from e

    nome_saida = Path(path_arquivo).stem + ".json"
    caminho_saida = os.path.join(OUTPUT_DIR, nome_saida)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    print(f"[OK] {path_arquivo} → {len(dados)} relações")


# =========================
# MAIN
# =========================

def main():

    parser = argparse.ArgumentParser(description="Extração de relações para grafo de conhecimento")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="API a utilizar: gemini (padrão) ou openai",
    )
    args = parser.parse_args()

    print(f"[INFO] Usando provider: {args.provider}")

    processados = carregar_processados()
    arquivos = Path(INPUT_DIR).glob("*.md")

    for arquivo in arquivos:

        nome = arquivo.name

        if processados.get(nome):
            print(f"[SKIP] {nome}")
            continue

        try:

            processar_arquivo(arquivo, args.provider)

            processados[nome] = True

            salvar_processados(processados)

        except Exception as e:

            print(f"[ERRO] {nome}")
            print(e)


if __name__ == "__main__":
    main()