import sys
import traceback

print("[INFO] Iniciando teste da interface LLM...")

# ── Carrega grafo ─────────────────────────────────────────────────────────────
try:
    from kg import KGRuntime, KGInterface
    print("[OK] Importação bem-sucedida")
except Exception:
    print("[ERRO] Falha ao importar kg:")
    traceback.print_exc()
    sys.exit(1)

try:
    rt = KGRuntime(verbose=True)
    print(f"[OK] Grafo carregado: {rt.graph.number_of_nodes()} nós / {rt.graph.number_of_edges()} arestas")
except Exception:
    print("[ERRO] Falha ao inicializar KGRuntime:")
    traceback.print_exc()
    sys.exit(1)

# ── Inicializa interface ──────────────────────────────────────────────────────
try:
    iface = KGInterface(rt.query, provider="openrouter", verbose=True)
    print("[OK] KGInterface inicializada (provider=openai)")
except Exception:
    print("[ERRO] Falha ao inicializar KGInterface:")
    traceback.print_exc()
    sys.exit(1)

# ── Perguntas de teste ────────────────────────────────────────────────────────
perguntas = [
    "Quantos nós e arestas tem o grafo?",
    "Qual é o nó mais central segundo o PageRank?",
    "Como Jeffrey Epstein se conecta a Ghislaine Maxwell?",
    "Quem são os intermediários estratégicos do grafo?",
]

for pergunta in perguntas:
    print(f"\n{'='*60}")
    print(f"Pergunta: {pergunta}")
    print(f"{'='*60}")
    try:
        resposta = iface.chat(pergunta)
        if resposta:
            print(f"Resposta:\n{resposta}")
        else:
            print("[AVISO] Resposta vazia — a LLM não retornou texto.")
    except Exception:
        print("[ERRO] Falha ao chamar iface.chat():")
        traceback.print_exc()

print("\n[INFO] Teste concluído.")
