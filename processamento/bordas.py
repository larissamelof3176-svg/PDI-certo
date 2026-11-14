"""
processamento/bordas.py
=======================
Detecção de bordas por Sobel — RF-08 do escopo.
Responsável: Pessoa 5 (P5).

Teoria
------
Uma "borda" é uma região onde a intensidade muda rápido. O gradiente mede
exatamente isso: aponta para onde a imagem fica mais clara, e seu MÓDULO diz
o quão brusca é a transição.

Sobel aproxima as derivadas parciais com duas correlações 3×3:

          ⎡-1  0  1⎤                ⎡-1 -2 -1⎤
    Gx =  ⎢-2  0  2⎥        Gy =    ⎢ 0  0  0⎥
          ⎣-1  0  1⎦                ⎣ 1  2  1⎦

Gx responde a variações na horizontal (bordas verticais) e Gy a variações na
vertical (bordas horizontais). Os pesos 1-2-1 nas linhas/colunas equivalem a
uma suavização embutida: por isso o Sobel é bem menos sensível a ruído do que
um simples "diferença entre vizinhos".

Magnitude (magnitude do gradiente):
    G = √(Gx² + Gy²)

Direção (não usada aqui, mas é o que o Canny aproveita depois):
    θ = arctan2(Gy, Gx)

Por que somar os quadrados em vez de |Gx| + |Gy|: a magnitude euclidiana é
invariante à rotação — uma borda inclinada 45° responde igual a uma vertical,
o que não acontece com a soma dos módulos. O preço é uma raiz quadrada.

Convenção de sinal: usamos correlação (mesma convenção do `cv2.Sobel`), então
o sinal de Gx/Gy é o do OpenCV. Como só usamos o módulo, isso não muda o
resultado — mas permite comparar os valores com o OpenCV nos testes.
"""

from __future__ import annotations

import numpy as np

from .filtros import convolucao
from .pontuais import escala_cinza, binarizar as _binarizar_pontual

# Kernels de Sobel (mesmos do OpenCV e do escopo)
GX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float64)

GY = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]], dtype=np.float64)


def gradiente_sobel(imagem: np.ndarray) -> tuple:
    """Calcula as duas derivadas parciais (Gx e Gy) por Sobel.

    A imagem é convertida para tons de cinza antes: em cor, cada canal teria
    suas próprias bordas e o resultado não seria mais "a borda da imagem", e
    sim três detecções sobrepostas.

    Devolve (gx, gy) em float64, COM sinal preservado — útil para inspecionar
    a direção do gradiente. A magnitude sai de `detectar_bordas`.
    """
    cinza = escala_cinza(imagem).astype(np.float64)
    return convolucao(cinza, GX), convolucao(cinza, GY)


def magnitude_gradiente(imagem: np.ndarray) -> np.ndarray:
    """Magnitude do gradiente G = √(Gx² + Gy²), em float64 sem normalizar.

    Os valores podem passar de 255 (uma borda perfeita preto→branco dá
    Gx ≈ 1020), por isso a normalização é um passo separado.
    """
    gx, gy = gradiente_sobel(imagem)
    return np.hypot(gx, gy)


def normalizar_para_8bits(magnitude: np.ndarray) -> np.ndarray:
    """Escala a magnitude para 0..255 proporcionalmente ao maior valor.

    Sem isso a imagem sairia quase toda preta (bordas típicas ficam bem abaixo
    de 1020) ou saturada. Uma imagem uniforme (magnitude toda zero) é devolvida
    como zeros, sem divisão por zero.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    maximo = float(magnitude.max()) if magnitude.size else 0.0
    if maximo <= 0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.clip(np.rint(magnitude * (255.0 / maximo)), 0, 255).astype(np.uint8)


def detectar_bordas(imagem: np.ndarray, normalizar: bool = True,
                    binarizar_resultado: bool = False, limiar: int = 128,
                    automatico: bool = False) -> np.ndarray:
    """Detecta bordas por Sobel e devolve uma imagem uint8 de um canal.

    Parâmetros
    ----------
    normalizar : se True (padrão), estica a magnitude para ocupar 0..255 — o
        mapa de bordas fica visível sem precisar ajustar nada. Se False, os
        valores crus do gradiente são apenas saturados em 255 (é o que o
        `cv2.Sobel` com CV_8U faz, útil para conferir a conta).
    binarizar_resultado : se True, converte em preto e branco (borda / não
        borda) usando o limiar. Útil para separar contornos de forma limpa.
    limiar : T do escopo, em 0..255 (usado só se `binarizar_resultado`).
    automatico : se True, calcula T pelo método de Otsu (ignora `limiar`).

    Retorna: uint8 (H, W) — o mapa de bordas.
    """
    magnitude = magnitude_gradiente(imagem)
    if normalizar:
        mapa = normalizar_para_8bits(magnitude)
    else:
        mapa = np.clip(np.rint(magnitude), 0, 255).astype(np.uint8)
    if binarizar_resultado:
        return _binarizar_pontual(mapa, limiar=limiar, automatico=automatico)
    return mapa


# Catálogo no mesmo formato de `pontuais.OPERACOES_PONTUAIS` e
# `filtros.OPERACOES_FILTROS`: a interface (P2) monta os controles daqui e o
# histórico (P1) guarda "nome + parâmetros" para reaplicar.
OPERACOES_BORDAS = {
    "detectar_bordas": {
        "funcao": detectar_bordas,
        "rotulo": "Detecção de bordas (Sobel)",
        "parametros": {
            "limiar": {"min": 0, "max": 255, "passo": 1, "padrao": 128},
            "binarizar_resultado": {"tipo": "bool", "padrao": False},
            "automatico": {"tipo": "bool", "padrao": False},
        },
    },
}


def descrever_operacao(nome: str, **parametros) -> str:
    """Texto curto para o histórico/relatório. Ex.: 'Detecção de bordas (Sobel)'."""
    info = OPERACOES_BORDAS.get(nome)
    rotulo = info["rotulo"] if info else nome
    if not parametros:
        return rotulo
    partes = ", ".join(
        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in parametros.items()
    )
    return f"{rotulo} ({partes})"
