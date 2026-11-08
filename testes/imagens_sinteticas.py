"""
Gerador de imagens sintéticas para os testes (não depende de arquivos externos).
Cobre os casos do banco de testes do escopo: escura, clara, baixo contraste,
ruído sal e pimenta, colorida, cinza e alta resolução.
"""

import numpy as np


def cena_base(altura: int = 256, largura: int = 256, semente: int = 0) -> np.ndarray:
    """Cena colorida 'equilibrada': gradientes, formas geométricas e textura suave."""
    rng = np.random.default_rng(semente)
    yy, xx = np.mgrid[0:altura, 0:largura].astype(np.float64)
    yy /= altura
    xx /= largura
    r = 40 + 170 * xx
    g = 40 + 170 * yy
    b = 40 + 170 * (1 - xx) * (1 - yy) + 60 * np.sin(6 * xx)
    img = np.stack([r, g, b], axis=-1)
    # Formas (bordas nítidas)
    img[(xx - 0.3) ** 2 + (yy - 0.35) ** 2 < 0.02] = [230, 220, 60]
    img[(xx > 0.55) & (xx < 0.85) & (yy > 0.55) & (yy < 0.85)] = [30, 40, 120]
    img[(np.abs(xx - yy) < 0.02)] = [250, 250, 250]
    # Textura fraca (não deve ser vista como ruído)
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def para_cinza(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img.astype(float) @ [0.299, 0.587, 0.114]), 0, 255).astype(np.uint8)


def escura(img: np.ndarray) -> np.ndarray:
    """Subexposição forte: escurece E comprime a faixa (escura + baixo contraste)."""
    return np.clip(img.astype(float) * 0.3, 0, 255).astype(np.uint8)


def clara(img: np.ndarray) -> np.ndarray:
    """Superexposição forte: clareia E comprime a faixa (clara + baixo contraste)."""
    return np.clip(255 - (255 - img.astype(float)) * 0.3, 0, 255).astype(np.uint8)


def escura_gama(img: np.ndarray, gama: float = 2.5) -> np.ndarray:
    """Escura mas ainda com contraste (só 'muito escura')."""
    return np.clip(np.rint(255 * (img / 255.0) ** gama), 0, 255).astype(np.uint8)


def clara_gama(img: np.ndarray, gama: float = 3.0) -> np.ndarray:
    """Clara mas ainda com contraste (só 'muito clara')."""
    return (255 - escura_gama(255 - img, gama)).astype(np.uint8)


def baixo_contraste(img: np.ndarray) -> np.ndarray:
    return np.clip((img.astype(float) - 128) * 0.25 + 128, 0, 255).astype(np.uint8)


def sal_e_pimenta(img: np.ndarray, densidade: float = 0.05, semente: int = 1) -> np.ndarray:
    rng = np.random.default_rng(semente)
    saida = img.copy()
    mascara = rng.random(img.shape[:2])
    saida[mascara < densidade / 2] = 0
    saida[mascara > 1 - densidade / 2] = 255
    return saida


def banco_de_teste(semente: int = 0) -> dict:
    """Devolve {nome: (imagem, conjunto_de_problemas_esperados)}."""
    base = cena_base(semente=semente)
    cinza = para_cinza(base)
    return {
        "equilibrada_cor": (base, set()),
        "equilibrada_cinza": (cinza, set()),
        "escura_cor": (escura_gama(base), {"muito escura"}),
        "escura_cinza": (escura_gama(cinza), {"muito escura"}),
        "clara_cor": (clara_gama(base), {"muito clara"}),
        "clara_cinza": (clara_gama(cinza), {"muito clara"}),
        "subexposta_cor": (escura(base), {"muito escura", "baixo contraste"}),
        "superexposta_cinza": (clara(cinza), {"muito clara", "baixo contraste"}),
        "escura_ruidosa_cinza": (sal_e_pimenta(escura_gama(cinza)), {"muito escura", "ruidosa"}),
        "baixo_contraste_cor": (baixo_contraste(base), {"baixo contraste"}),
        "baixo_contraste_cinza": (baixo_contraste(cinza), {"baixo contraste"}),
        "ruidosa_cor": (sal_e_pimenta(base), {"ruidosa"}),
        "ruidosa_cinza": (sal_e_pimenta(cinza), {"ruidosa"}),
    }
