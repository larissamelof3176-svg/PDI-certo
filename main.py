"""
main.py
=======
Ponto de entrada do Sistema de Análise, Processamento e Melhoria de Imagens
Digitais. Responsável: Pessoa 1 (P1).

Enquanto a interface gráfica (P2) não existe, este é um executável de LINHA DE
COMANDO que roda o fluxo completo do escopo:

    abrir → analisar → processar → medir → salvar → relatar

Ele não duplica lógica: só costura `nucleo` (I/O e histórico), `processamento`
(P4/P5) e `analise` (P3). Com isso serve para dois papéis:

1. Smoke test de integração da seção 12.2 — se `python main.py <imagem>` roda
   do começo ao fim, os módulos das cinco pessoas estão compatíveis.
2. Uma forma de usar o sistema antes da GUI ficar pronta. A interface gráfica
   vai reaproveitar exatamente as mesmas chamadas.

Uso
---
    python main.py imagem.jpg
    python main.py imagem.png --saida resultados --pipeline
    python main.py imagem.bmp --pipeline --bordas

    --saida PASTA   onde gravar resultado e relatório (padrão: resultados/)
    --pipeline      usa a cadeia fixa do escopo (cinza → equalizar → mediana)
                    em vez da melhoria automática
    --bordas        acrescenta a detecção de bordas (Sobel) ao final
    --sem-relatorio não gera os arquivos de relatório (só o resultado)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from nucleo import ErroImagem, Historico, abrir_imagem, salvar_imagem, sugerir_nome_resultado
from processamento import (
    automatico,
    equalizacao,
    escala_cinza,
    filtro_mediana,
    detectar_bordas,
)

try:
    # Informações básicas são RF-03, da P3 — o núcleo não as reimplementa.
    from analise.metricas import informacoes_basicas
    from analise.relatorio import gerar_relatorio
except ImportError as erro:  # matplotlib/fpdf2 são opcionais para processar
    informacoes_basicas = None
    gerar_relatorio = None
    _MOTIVO_SEM_RELATORIO = str(erro)


# ---------------------------------------------------------------------------
# Saída no terminal
# ---------------------------------------------------------------------------
def _titulo(texto: str) -> None:
    print(f"\n{texto}")
    print("-" * len(texto))


def _imprimir_informacoes(imagem: np.ndarray, caminho: Path) -> None:
    _titulo("1. Imagem")
    print(f"Arquivo : {caminho.name}")

    if informacoes_basicas is None:
        # Sem a camada de análise instalada, mostramos o que dá para tirar do
        # próprio array — o processamento em si não depende da P3.
        altura, largura = imagem.shape[:2]
        canais = 1 if imagem.ndim == 2 else imagem.shape[2]
        print(f"Dimensoes: {largura} x {altura} px")
        print(f"Canais   : {canais}   Pixels: {largura * altura}")
        return

    info = informacoes_basicas(imagem, caminho)
    if info["formato"]:
        print(f"Formato : {info['formato']}  ({info['tamanho_kb']} KB)")
    print(f"Dimensoes: {info['largura']} x {info['altura']} px")
    print(f"Canais   : {info['canais']}   Pixels: {info['total_pixels']:,}".replace(",", "."))


def _imprimir_diagnostico(imagem: np.ndarray, rotulo: str) -> None:
    _titulo(rotulo)
    print(automatico.diagnosticar(imagem).texto())


# ---------------------------------------------------------------------------
# Fluxos
# ---------------------------------------------------------------------------
def fluxo_automatico(imagem: np.ndarray) -> tuple:
    """Melhoria automática (RF-09), empilhando cada passo no histórico.

    Devolve (imagem_final, historico, justificativa, rotulo_do_arquivo).
    O rótulo é "melhoria automatica" e não o nome do último passo: a decisão
    foi do sistema inteiro, e nomear o arquivo pela última correção aplicada
    ("correcao_gama") esconderia as outras.
    """
    resultado = automatico.melhorar_automaticamente(imagem)

    historico = Historico(imagem)
    historico.empilhar_melhoria(resultado)

    _titulo("3. Melhoria automatica")
    for linha in resultado.justificativa().splitlines():
        print(f"  {linha}")

    return resultado.imagem, historico, resultado.justificativa(), "melhoria automatica"


def fluxo_pipeline(imagem: np.ndarray, com_bordas: bool) -> tuple:
    """Cadeia fixa do escopo (seção 12.2): cinza → equalizar → mediana [→ bordas].

    Diferente da melhoria automática, aqui as etapas são impostas — é o teste
    de que TODAS as operações funcionam encadeadas, e não só as que as regras
    de diagnóstico escolheriam.

    Devolve (imagem_final, historico, justificativa, rotulo_do_arquivo).
    """
    etapas = [
        ("Escala de cinza", escala_cinza, {}),
        ("Equalização de histograma", equalizacao.equalizar_histograma, {}),
        ("Filtro da mediana 3x3", filtro_mediana, {"tamanho": 3}),
    ]
    if com_bordas:
        etapas.append(("Detecção de bordas (Sobel)", detectar_bordas, {}))

    historico = Historico(imagem)
    _titulo("3. Pipeline (cadeia fixa)")
    for rotulo, funcao, parametros in etapas:
        nova = funcao(historico.imagem_atual, **parametros)
        historico.empilhar(rotulo, nova, **parametros)
        print(f"  {len(historico.operacoes)}. {rotulo}")

    justificativa = (
        "Pipeline de teste do escopo, aplicado em sequência:\n  "
        + "\n  ".join(f"{i}. {r}" for i, r in enumerate(historico.operacoes, 1))
    )
    return historico.imagem_atual, historico, justificativa, historico.rotulo_atual


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Sistema de Análise, Processamento e Melhoria de Imagens Digitais.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("imagem", help="caminho da imagem (JPG, JPEG, PNG ou BMP)")
    parser.add_argument("--saida", default="resultados",
                        help="pasta para o resultado e o relatório (padrão: resultados)")
    parser.add_argument("--pipeline", action="store_true",
                        help="usa a cadeia fixa (cinza → equalizar → mediana)")
    parser.add_argument("--bordas", action="store_true",
                        help="acrescenta a detecção de bordas (Sobel)")
    parser.add_argument("--sem-relatorio", action="store_true",
                        help="não gera o relatório .txt/.pdf")
    args = parser.parse_args(argv)

    print("=" * 62)
    print("Sistema de Analise, Processamento e Melhoria de Imagens Digitais")
    print("=" * 62)

    try:
        original = abrir_imagem(args.imagem)
    except ErroImagem as erro:
        print(f"\nERRO: {erro}", file=sys.stderr)
        return 1

    caminho = Path(args.imagem)
    pasta = Path(args.saida)

    _imprimir_informacoes(original, caminho)
    _imprimir_diagnostico(original, "2. Diagnostico inicial")

    if args.pipeline:
        processada, historico, justificativa, rotulo_saida = fluxo_pipeline(original, args.bordas)
    else:
        if args.bordas:
            print("\nAviso: --bordas so tem efeito junto com --pipeline. Ignorando.")
        processada, historico, justificativa, rotulo_saida = fluxo_automatico(original)

    _imprimir_diagnostico(processada, f"4. Diagnostico final (apos {len(historico.operacoes)} operacao(oes))")

    destino = sugerir_nome_resultado(caminho, rotulo_saida or "resultado", pasta)
    try:
        salvar_imagem(processada, destino)
    except ErroImagem as erro:
        print(f"\nERRO ao salvar: {erro}", file=sys.stderr)
        return 1
    _titulo("5. Resultado")
    print(f"Imagem salva em: {destino}")

    if args.sem_relatorio:
        return 0
    if gerar_relatorio is None:
        print(f"\nAviso: relatorio nao gerado (dependencia ausente: {_MOTIVO_SEM_RELATORIO}).")
        print("       Instale com: pip install -r requirements.txt")
        return 0

    try:
        arquivo_txt, arquivo_pdf = gerar_relatorio(
            nome=caminho.name,
            original=original,
            processada=processada,
            operacoes=historico.operacoes,
            justificativa=justificativa,
            caminho=caminho,
            pasta=pasta,
        )
    except Exception as erro:  # relatório é o último passo: não derruba o resto
        print(f"\nAviso: nao foi possivel gerar o relatorio ({erro}).")
        return 0

    print(f"Relatorio .txt : {arquivo_txt}")
    print(f"Relatorio .pdf : {arquivo_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
