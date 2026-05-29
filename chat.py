"""
Loop interativo de consulta ao Knowledge Graph via LLM.

Uso:
    python chat.py
    python chat.py --provider openrouter
    python chat.py --provider anthropic --model claude-opus-4-7
    python chat.py --verbose

Comandos durante a sessão:
    /reset    — limpa o histórico de conversa
    /stats    — exibe estatísticas do grafo
    /modelo   — exibe provider e modelo em uso
    sair      — encerra
"""

import argparse
import sys
import traceback

from kg import KGRuntime, KGInterface

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

PROVIDERS = ("openrouter", "openai", "anthropic", "gemini")

DEFAULT_MODELS = {
    "openrouter": "google/gemma-4-31b-it:free",
    "openai":     "gpt-4o",
    "anthropic":  "claude-opus-4-7",
    "gemini":     "gemini-2.0-flash",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat interativo com o Knowledge Graph via LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="openrouter",
        help=f"Provider LLM (padrão: openrouter)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo a usar (usa o padrão do provider se omitido)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe chamadas de ferramenta e dados retornados",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def init(args: argparse.Namespace) -> tuple[KGRuntime, KGInterface]:
    print("Carregando grafo...", end=" ", flush=True)
    try:
        rt = KGRuntime(verbose=False)
    except Exception:
        print("\n[ERRO] Falha ao carregar o grafo:")
        traceback.print_exc()
        sys.exit(1)

    n = rt.graph.number_of_nodes()
    e = rt.graph.number_of_edges()
    print(f"{n} nós / {e} arestas")

    model = args.model or DEFAULT_MODELS[args.provider]
    print(f"Provider : {args.provider}")
    print(f"Modelo   : {model}")

    try:
        iface = KGInterface(rt.query, provider=args.provider, model=model, verbose=args.verbose)
    except Exception:
        print("\n[ERRO] Falha ao inicializar a interface LLM:")
        traceback.print_exc()
        sys.exit(1)

    return rt, iface


# ─────────────────────────────────────────────────────────────────────────────
# COMANDOS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def handle_command(cmd: str, rt: KGRuntime, iface: KGInterface) -> bool:
    """Retorna True se o comando foi reconhecido e tratado."""
    if cmd == "/reset":
        iface.reset()
        print("[histórico limpo]")
        return True

    if cmd == "/stats":
        stats = rt.query.get_graph_stats()
        print(f"Nós        : {stats['nodes']}")
        print(f"Arestas    : {stats['edges']}")
        print(f"Comunidades: {stats['communities']}")
        print(f"Densidade  : {stats['density']:.6f}")
        print(f"Relações   : {', '.join(r for r, _ in stats['top_relations'][:5])}")
        return True

    if cmd == "/modelo":
        print(f"Provider : {iface._provider}")
        print(f"Modelo   : {iface._model}")
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def loop(rt: KGRuntime, iface: KGInterface) -> None:
    print("\nDigite sua pergunta. Comandos: /reset  /stats  /modelo  sair\n")

    while True:
        try:
            entrada = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not entrada:
            continue

        if entrada.lower() in ("sair", "exit", "quit", "q"):
            print("Encerrando.")
            break

        if entrada.startswith("/"):
            if not handle_command(entrada, rt, iface):
                print(f"Comando desconhecido: {entrada}")
            continue

        try:
            resposta = iface.chat(entrada)
            print(f"\n{resposta}\n")
        except KeyboardInterrupt:
            print("\n[interrompido]")
        except Exception as e:
            print(f"[ERRO] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    rt, iface = init(args)
    loop(rt, iface)
