"""Relatório automático .txt e .pdf (P3 - RF-12)."""
import io
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from fpdf import FPDF

from analise.metricas import metricas, diagnostico, informacoes_basicas, comparar
from analise.histograma import figura_comparacao

PASTA_PADRAO = Path("resultados")


def _texto_relatorio(nome, original, processada, operacoes, justificativa=None,
                     caminho=None) -> str:
    info = informacoes_basicas(original, caminho)
    linhas = [
        "RELATORIO DE PROCESSAMENTO DE IMAGEM",
        f"Gerado em: {datetime.now():%d/%m/%Y %H:%M}",
        "",
        "1. IMAGEM",
        f"Nome: {nome}",
        f"Dimensoes: {info['largura']} x {info['altura']} px",
        f"Canais: {info['canais']}   Total de pixels: {info['total_pixels']}",
    ]
    if info["formato"]:
        linhas.append(
            f"Formato: {info['formato']}   Tamanho: {info['tamanho_kb']} KB")
    linhas += ["", "2. DIAGNOSTICO INICIAL", diagnostico(original), "",
               "3. OPERACOES APLICADAS"]
    if operacoes:
        linhas += [f"{i}. {op}" for i, op in enumerate(operacoes, 1)]
    else:
        linhas.append("Nenhuma operacao aplicada.")
    if justificativa:
        linhas += ["", "4. JUSTIFICATIVA DA MELHORIA AUTOMATICA", justificativa]
    linhas += ["", "5. METRICAS ANTES / DEPOIS",
               f"{'metrica':<12}{'antes':>10}{'depois':>10}{'delta':>10}"]
    for r in comparar(original, processada):
        linhas.append(f"{r['metrica']:<12}{r['antes']:>10.2f}"
                      f"{r['depois']:>10.2f}{r['delta']:>+10.2f}")
    linhas += ["", f"Diagnostico final: {diagnostico(processada)}"]
    return "\n".join(linhas)


def gerar_relatorio_txt(nome, original, processada, operacoes,
                        justificativa=None, caminho=None, pasta=PASTA_PADRAO) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "relatorio.txt"
    destino.write_text(
        _texto_relatorio(nome, original, processada,
                         operacoes, justificativa, caminho),
        encoding="utf-8")
    return destino


def _png(arr, largura_max=500) -> io.BytesIO:
    img = Image.fromarray(np.asarray(arr))
    img.thumbnail((largura_max, largura_max))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


_TROCAS = {
    "γ": "gama", "σ": "sigma", "α": "alfa", "Δ": "delta",
    "–": "-", "—": "-", "→": "->", "≈": "~", "≥": ">=", "≤": "<=",
    "×": "x", "−": "-",
}


def _latin1(s: str) -> str:
    for k, v in _TROCAS.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def gerar_relatorio_pdf(nome, original, processada, operacoes,
                        justificativa=None, caminho=None, pasta=PASTA_PADRAO) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "relatorio.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    texto = _texto_relatorio(nome, original, processada, operacoes,
                             justificativa, caminho)
    pdf.set_font("Courier", size=9)
    pdf.multi_cell(0, 5, _latin1(texto), new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Imagens: original e processada",
             new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y() + 2
    pdf.image(_png(original), x=10, y=y, w=90)
    pdf.image(_png(processada), x=105, y=y, w=90)

    fig = figura_comparacao(original, processada)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    pdf.image(buf, x=10, y=y + 100, w=185)

    pdf.output(str(destino))
    return destino


def gerar_relatorio(nome, original, processada, operacoes,
                    justificativa=None, caminho=None, pasta=PASTA_PADRAO):
    """Gera .txt e .pdf. Devolve (caminho_txt, caminho_pdf)."""
    return (
        gerar_relatorio_txt(nome, original, processada, operacoes,
                            justificativa, caminho, pasta),
        gerar_relatorio_pdf(nome, original, processada, operacoes,
                            justificativa, caminho, pasta),
    )
