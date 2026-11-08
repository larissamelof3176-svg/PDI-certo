"""Histogramas de 256 níveis (P3 - RF-04)."""
import numpy as np
from matplotlib.figure import Figure
from analise.metricas import luminancia


def histograma(img) -> np.ndarray:
    """Histograma da luminância, 256 posições, contado manualmente."""
    y = np.clip(np.rint(luminancia(img)), 0, 255).astype(np.intp)
    return np.bincount(y.ravel(), minlength=256)


def histograma_canais(img) -> dict:
    """Um histograma por canal (R, G, B). Cinza devolve só 'Y'."""
    img = np.asarray(img)
    if img.ndim == 2:
        return {"Y": histograma(img)}
    return {
        nome: np.bincount(img[..., i].ravel(), minlength=256)
        for i, nome in enumerate("RGB")
    }


def desenhar_histograma(ax, img, titulo=""):
    """Desenha o histograma em um Axes do Matplotlib."""
    ax.bar(range(256), histograma(img), width=1, color="gray")
    ax.set_xlim(0, 255)
    ax.set_title(titulo, fontsize=9)
    ax.set_xlabel("Intensidade")
    ax.set_ylabel("Pixels")


def figura_comparacao(antes, depois) -> Figure:
    """Figura com os dois histogramas lado a lado (para embutir no Tkinter)."""
    fig = Figure(figsize=(8, 3))
    desenhar_histograma(fig.add_subplot(1, 2, 1), antes, "Original")
    desenhar_histograma(fig.add_subplot(1, 2, 2), depois, "Processada")
    fig.tight_layout()
    return fig