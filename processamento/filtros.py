"""
processamento/filtros.py
========================
Filtros espaciais no domínio do espaço — RF-07 do escopo.
Responsável: Pessoa 5 (P5).

Teoria
------
Um filtro espacial percorre a imagem com uma janela (kernel) e substitui o
pixel central por uma combinação dos vizinhos. Para uma janela 3×3:

    g(x, y) = Σ_{i,j} w(i, j) · f(x + i, y + j)

Essa conta é uma CORRELAÇÃO. A convolução "de verdade" espelha o kernel antes
(equivalente a girar a janela 180°); para kernels simétricos — média, mediana,
gaussiano, laplaciano — os dois resultados são idênticos. Seguimos a mesma
convenção do OpenCV (`filter2D`/`Sobel`), que também usa correlação, para que
os testes de validação comparem valores iguais.

Bordas: o escopo pede replicação de pixels (padding 'edge') — o pixel da borda
é repetido para fora da imagem. Sem isso a janela sairia da imagem e faltariam
vizinhos para os pixels das bordas.

Vetorização: em vez de um laço por pixel (que travaria a interface), usamos
`sliding_window_view`, que devolve TODAS as janelas da imagem como uma única
visão (sem copiar memória), e um `einsum` que multiplica todas de uma vez pelo
kernel. É a mesma técnica já usada em `automatico.py` para estimar o ruído.

Contrato
--------
Todas as funções públicas seguem o padrão do projeto:
    funcao(imagem: np.ndarray, **parametros) -> np.ndarray
- Aceitam imagens em tons de cinza (H, W) ou coloridas (H, W, 3) / (H, W, 4).
- Nunca modificam o array de entrada (funções puras).
- Sempre devolvem uint8 com valores em 0..255 (saturação com np.clip).
- O canal alfa (RGBA) é preservado — não faz sentido "borrar" a transparência.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .pontuais import _validar_imagem, _separar_alfa, _juntar_alfa

# Tamanhos de janela aceitos por cada filtro (RF-07)
TAMANHOS_MEDIA = (3, 5, 7)
TAMANHOS_MEDIANA = (3, 5)

# Kernel laplaciano (centro positivo): realça variações de intensidade.
# Ver `nitidez()` para a relação com o laplaciano de centro negativo.
LAPLACIANO_4 = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Núcleo: convolução vetorizada
# ---------------------------------------------------------------------------
def _validar_kernel(kernel) -> np.ndarray:
    """Confere que o kernel é uma matriz quadrada de lado ímpar (exigência para
    que exista um pixel central)."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"O kernel deve ser uma matriz quadrada; recebido {kernel.shape}.")
    if kernel.shape[0] % 2 == 0:
        raise ValueError(
            f"O lado do kernel deve ser ímpar (3, 5, 7...) para haver um pixel "
            f"central; recebido {kernel.shape[0]}."
        )
    if kernel.shape[0] < 1:
        raise ValueError("O kernel está vazio.")
    return kernel


def _convolver_canal(canal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlação de UM canal (H, W) com o kernel, com padding por replicação.

    Devolve float64 SEM saturar: quem chama decide (Sobel precisa dos valores
    negativos do gradiente; um filtro de média já sabe que quer 0..255).
    """
    raio = kernel.shape[0] // 2
    # np.pad com mode="edge" = replicação de pixels (o escopo pede exatamente isso)
    preenchido = np.pad(canal, raio, mode="edge")
    # (H, W, t, t): todas as janelas da imagem, sem copiar dados
    janelas = sliding_window_view(preenchido, kernel.shape)
    return np.einsum("ijkl,kl->ij", janelas, kernel)


def convolucao(imagem, kernel) -> np.ndarray:
    """Aplica `kernel` à imagem inteira e devolve float64 (sem saturar).

    Aceita (H, W), (H, W, 3) ou (H, W, 4) e devolve a mesma quantidade de
    canais em float64. Um formato (H, W, 1) é achatado para (H, W) — é a mesma
    convenção do `_validar_imagem` da P4, que trata canal único como cinza.

    É a base de todos os filtros deste módulo e também do Sobel em `bordas.py`.

    Função "crua" de propósito: não converte para uint8 nem satura, porque o
    detector de bordas precisa enxergar respostas NEGATIVAS do gradiente (que
    um clip em 0..255 destruiria). Os filtros públicos abaixo fecham o ciclo.
    """
    if imagem is None:
        raise ValueError("Nenhuma imagem foi fornecida.")
    dados = np.asarray(imagem, dtype=np.float64)
    if dados.ndim not in (2, 3):
        raise ValueError(f"Formato não suportado para filtragem: {dados.shape}.")
    if dados.size == 0:
        raise ValueError("A imagem está vazia.")
    kernel = _validar_kernel(kernel)

    if dados.ndim == 2:
        return _convolver_canal(dados, kernel)
    return np.stack([_convolver_canal(dados[:, :, c], kernel)
                     for c in range(dados.shape[2])], axis=-1)


def _aplicar_em_float(imagem, operacao: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Roda `operacao` sobre os canais de cor em float64 e devolve uint8.

    Concentra o tratamento repetido em todos os filtros: validar a entrada,
    tirar o alfa, trabalhar em float, saturar de volta para 0..255 e recolocar
    o alfa. Evita que cada filtro repita (e erre) esse ciclo.
    """
    imagem = _validar_imagem(imagem)
    rgb, alfa = _separar_alfa(imagem)
    saida = np.clip(np.rint(operacao(rgb.astype(np.float64))), 0, 255).astype(np.uint8)
    return _juntar_alfa(saida, alfa)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def kernel_media(tamanho: int = 3) -> np.ndarray:
    """Kernel da média: todos os pesos iguais a 1/(t·t).

    Como a soma dos pesos é 1, a média preserva o brilho médio da imagem — é
    por isso que ela funciona como suavização sem alterar a exposição.
    """
    if tamanho not in TAMANHOS_MEDIA:
        raise ValueError(f"O tamanho do filtro da média deve ser um de {TAMANHOS_MEDIA}.")
    return np.full((tamanho, tamanho), 1.0 / (tamanho * tamanho), dtype=np.float64)


def kernel_gaussiano(tamanho: int = 3, sigma: float | None = None) -> np.ndarray:
    """Kernel gaussiano 2D normalizado.

    Usado pelo unsharp masking: um borrão gaussiano pesa mais o pixel central
    e cai suavemente, então a máscara de detalhe (f − suave) fica sem os
    "anéis" que a média produz.
    """
    if tamanho not in TAMANHOS_MEDIA:
        raise ValueError(f"O tamanho do kernel gaussiano deve ser um de {TAMANHOS_MEDIA}.")
    if sigma is None:
        sigma = tamanho / 3.0
    eixo = np.arange(tamanho, dtype=np.float64) - (tamanho - 1) / 2.0
    g = np.exp(-(eixo[:, None] ** 2 + eixo[None, :] ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


# ---------------------------------------------------------------------------
# RF-07 · Filtro da média
# ---------------------------------------------------------------------------
def filtro_media(imagem: np.ndarray, tamanho: int = 3) -> np.ndarray:
    """Suaviza a imagem pela média dos vizinhos (kernel t×t uniforme).

    Efeito: reduz ruído aleatório, mas borra as bordas — a média não distingue
    "ruído" de "detalhe", apenas espalha tudo. Quanto maior a janela, maior a
    suavização e a perda de nitidez (3×3, 5×5 ou 7×7).

    Retorna: uint8 com o mesmo formato da entrada.
    """
    return _aplicar_em_float(imagem, lambda x: convolucao(x, kernel_media(tamanho)))


# ---------------------------------------------------------------------------
# RF-07 · Filtro da mediana
# ---------------------------------------------------------------------------
def filtro_mediana(imagem: np.ndarray, tamanho: int = 3) -> np.ndarray:
    """Remove ruído "sal e pimenta" preservando as bordas.

    Para cada pixel, ordena a vizinhança t×t e usa o valor do MEIO (mediana).
    Impulsos isolados (0 ou 255) nunca ficam no meio de uma vizinhança
    ordenada, então desaparecem — já uma borda, que é formada por MUITOS
    pixels da mesma faixa, sobrevive à ordenação. É essa diferença que faz a
    mediana ser melhor que a média para ruído impulsivo.

    IMPORTANTE: esta é a função que `automatico.py` procura importar
    (`from .filtros import filtro_mediana`); a assinatura `(imagem, tamanho)`
    é o contrato combinado entre a P4 e a P5.

    Retorna: uint8 com o mesmo formato da entrada.
    """
    if tamanho not in TAMANHOS_MEDIANA:
        raise ValueError(f"O tamanho do filtro da mediana deve ser um de {TAMANHOS_MEDIANA}.")
    imagem = _validar_imagem(imagem)
    rgb, alfa = _separar_alfa(imagem)
    raio = tamanho // 2

    def _canal(c: np.ndarray) -> np.ndarray:
        preenchido = np.pad(c, raio, mode="edge")
        janelas = sliding_window_view(preenchido, (tamanho, tamanho))
        janelas = janelas.reshape(c.shape[0], c.shape[1], -1)
        meio = janelas.shape[-1] // 2
        # partition é O(n) e garante que a posição `meio` já está ordenada
        return np.partition(janelas, meio, axis=-1)[..., meio].astype(np.uint8)

    if rgb.ndim == 2:
        return _canal(rgb)
    saida = np.stack([_canal(rgb[:, :, i]) for i in range(rgb.shape[2])], axis=-1)
    return _juntar_alfa(saida, alfa)


# ---------------------------------------------------------------------------
# RF-07 · Nitidez (sharpening)
# ---------------------------------------------------------------------------
METODOS_NITIDEZ = ("unsharp", "laplaciano")


def nitidez(imagem: np.ndarray, metodo: str = "unsharp", intensidade: float = 1.0,
            tamanho: int = 3) -> np.ndarray:
    """Realça os detalhes (bordas) da imagem.

    Parâmetros
    ----------
    metodo : 'unsharp' (padrão) ou 'laplaciano'.
        'unsharp'    → g = f + k·(f − suave). A diferença entre a imagem e uma
                       versão borrada (gaussiana) é justamente o "detalhe fino";
                       somá-la de volta realça as bordas. Vem da fotografia
                       analógica (máscara desfocada) e é o método mais estável.
        'laplaciano' → g = f + k·∇²f. O laplaciano destaca as regiões de
                       variação rápida (bordas); somá-lo à imagem acentua essas
                       transições. É mais agressivo e amplifica ruído.
    intensidade : quanto de detalhe somar. 0 → imagem original; 1 → realce
                  padrão; > 1 → realce forte (aparecem halos se exagerar).
    tamanho : janela do borrão gaussiano (só no método 'unsharp').

    O realce é aplicado a cada canal de cor. Em excesso, o detalhe somado pode
    estourar em 0 ou 255 — a saturação da `_aplicar_em_float` cuida disso.

    Retorna: uint8 com o mesmo formato da entrada.
    """
    metodo = metodo.lower()
    if metodo not in METODOS_NITIDEZ:
        raise ValueError(f"Método de nitidez inválido: {metodo}. Use um de {METODOS_NITIDEZ}.")
    if intensidade < 0:
        raise ValueError("A intensidade do realce não pode ser negativa.")

    if metodo == "unsharp":
        suave = kernel_gaussiano(tamanho)
        return _aplicar_em_float(
            imagem,
            lambda x: x + intensidade * (x - convolucao(x, suave)),
        )

    lap = np.asarray(LAPLACIANO_4)
    # LAPLACIANO_4 tem centro +4: somar k·∇²f realça a borda com o sinal certo.
    return _aplicar_em_float(imagem, lambda x: x + intensidade * convolucao(x, lap))


# ---------------------------------------------------------------------------
# Catálogo para a interface (P2) e o histórico (P1)
# ---------------------------------------------------------------------------
# Mesmo formato de `pontuais.OPERACOES_PONTUAIS`: a interface monta os sliders
# a partir daqui e o histórico guarda "nome + parâmetros" para reaplicar.
OPERACOES_FILTROS = {
    "filtro_media": {
        "funcao": filtro_media,
        "rotulo": "Filtro da média",
        "parametros": {"tamanho": {"min": 3, "max": 7, "passo": 2, "padrao": 3}},
    },
    "filtro_mediana": {
        "funcao": filtro_mediana,
        "rotulo": "Filtro da mediana",
        "parametros": {"tamanho": {"min": 3, "max": 5, "passo": 2, "padrao": 3}},
    },
    "nitidez": {
        "funcao": nitidez,
        "rotulo": "Nitidez (unsharp)",
        "parametros": {
            "intensidade": {"min": 0.0, "max": 3.0, "passo": 0.1, "padrao": 1.0},
            "tamanho": {"min": 3, "max": 7, "passo": 2, "padrao": 3},
        },
    },
}


def descrever_operacao(nome: str, **parametros) -> str:
    """Texto curto para o histórico/relatório. Ex.: 'Filtro da mediana (tamanho=3)'."""
    info = OPERACOES_FILTROS.get(nome)
    rotulo = info["rotulo"] if info else nome
    if not parametros:
        return rotulo
    partes = ", ".join(
        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in parametros.items()
    )
    return f"{rotulo} ({partes})"
