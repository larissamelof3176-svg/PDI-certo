"""Métricas e diagnóstico de imagem (P3 - RF-03 e RF-10)."""
import os
import numpy as np
from PIL import Image


def luminancia(img):
    """Y = 0,299R + 0,587G + 0,114B (float). Cinza passa direto."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.float64)
    r, g, b = (img[..., i].astype(np.float64) for i in range(3))
    return 0.299 * r + 0.587 * g + 0.114 * b


def metricas(img) -> dict:
    """Brilho médio, contraste (desvio padrão), mín, máx e amplitude."""
    y = luminancia(img)
    minimo, maximo = int(round(y.min())), int(round(y.max()))
    return {
        "brilho": float(y.mean()),
        "contraste": float(y.std()),
        "minimo": minimo,
        "maximo": maximo,
        "amplitude": maximo - minimo,
    }


def informacoes_basicas(img, caminho=None) -> dict:
    """Formato, largura, altura, canais, total de pixels, tamanho do arquivo."""
    img = np.asarray(img)
    altura, largura = img.shape[:2]
    info = {
        "formato": None,
        "largura": int(largura),
        "altura": int(altura),
        "canais": 1 if img.ndim == 2 else int(img.shape[2]),
        "total_pixels": int(largura * altura),
        "tamanho_kb": None,
    }
    if caminho:
        info["tamanho_kb"] = round(os.path.getsize(caminho) / 1024, 1)
        with Image.open(caminho) as f:
            info["formato"] = f.format
    return info


def diagnostico(img) -> str:
    """Rótulo textual. Usa o diagnóstico da P4 para não haver dois diferentes."""
    try:
        from processamento import automatico as A
        return A.diagnosticar(img).rotulo
    except Exception:
        m = metricas(img)
        if m["brilho"] < 85:
            return "muito escura"
        if m["brilho"] > 170:
            return "muito clara"
        if m["contraste"] < 35:
            return "baixo contraste"
        return "equilibrada"


def comparar(antes, depois) -> list:
    """Tabela antes/depois com a variação (Δ) de cada métrica."""
    ma, md = metricas(antes), metricas(depois)
    return [
        {"metrica": k, "antes": ma[k], "depois": md[k], "delta": md[k] - ma[k]}
        for k in ma
    ]