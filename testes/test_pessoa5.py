"""
Testes automatizados dos módulos da Pessoa 5 (pytest).
Cobre a seção 12.1 (testes individuais – Filtros e Bordas) e os critérios da
seção 13 ("Mediana remove >= 80% do ruído sal e pimenta", "nitidez realça
detalhes sem artefatos excessivos", "contornos principais visíveis").

Executar na raiz do projeto:  python -m pytest testes/ -v
"""

import time

import numpy as np
import pytest

from processamento import filtros as F
from processamento import bordas as B
from testes.imagens_sinteticas import (
    cena_base, para_cinza, sal_e_pimenta, baixo_contraste,
)

try:
    import cv2
except ImportError:  # OpenCV é opcional: só valida (seção 7 do escopo)
    cv2 = None


@pytest.fixture(scope="module")
def cor():
    return cena_base()


@pytest.fixture(scope="module")
def cinza(cor):
    return para_cinza(cor)


# ======================================================================
# Contrato geral (mesmo contrato dos módulos da P4)
# ======================================================================
FUNCOES = [
    (F.filtro_media, {}),
    (F.filtro_media, {"tamanho": 5}),
    (F.filtro_mediana, {}),
    (F.filtro_mediana, {"tamanho": 5}),
    (F.nitidez, {}),
    (F.nitidez, {"metodo": "laplaciano", "intensidade": 0.5}),
    (B.detectar_bordas, {}),
    (B.detectar_bordas, {"binarizar_resultado": True, "limiar": 60}),
]


@pytest.mark.parametrize("func,params", FUNCOES)
@pytest.mark.parametrize("tipo", ["cor", "cinza"])
def test_contrato_uint8_e_funcao_pura(func, params, tipo, cor, cinza):
    img = cor if tipo == "cor" else cinza
    copia = img.copy()
    out = func(img, **params)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape[:2] == img.shape[:2]
    assert np.array_equal(img, copia), "a função não pode alterar a entrada"


def test_rgba_preserva_alfa(cor):
    alfa = np.full(cor.shape[:2] + (1,), 77, np.uint8)
    rgba = np.concatenate([cor, alfa], axis=2)
    for f in (F.filtro_media, F.filtro_mediana, F.nitidez, B.detectar_bordas):
        out = f(rgba)
        # Bordas devolve 1 canal (mapa de intensidade); os filtros mantêm o alfa
        if out.ndim == 3 and out.shape[2] == 4:
            assert np.all(out[:, :, 3] == 77)


def test_entrada_invalida(cor):
    with pytest.raises(ValueError):
        F.filtro_media(None)
    with pytest.raises(ValueError):
        F.filtro_media(cor, tamanho=4)          # tamanho fora de {3,5,7}
    with pytest.raises(ValueError):
        F.filtro_mediana(cor, tamanho=7)        # mediana só aceita {3,5}
    with pytest.raises(ValueError):
        F.nitidez(cor, metodo="canny")          # método inexistente
    with pytest.raises(ValueError):
        F.nitidez(cor, intensidade=-1)          # intensidade negativa
    with pytest.raises(ValueError):
        F.convolucao(cor, np.ones((2, 2)))      # kernel de lado par


# ======================================================================
# Convolução (núcleo)
# ======================================================================
def test_convolucao_identidade_e_padding_por_replicacao():
    img = np.arange(9, dtype=np.float64).reshape(3, 3)
    identidade = np.zeros((3, 3))
    identidade[1, 1] = 1.0
    assert np.allclose(F.convolucao(img, identidade), img)

    # Padding 'edge' = replicação: o canto superior esquerdo da média 3x3 deve
    # ser a média da vizinhança com os pixels da borda repetidos.
    out = F.convolucao(img, np.full((3, 3), 1 / 9))
    vizinhanca = np.array([[0, 0, 1], [0, 0, 1], [3, 3, 4]], dtype=np.float64)
    assert np.isclose(out[0, 0], vizinhanca.mean())


def test_convolucao_preserva_canais(cor):
    out = F.convolucao(cor, F.kernel_media(3))
    assert out.shape == cor.shape
    assert out.dtype == np.float64     # cru: quem chama decide a saturação


def test_kernel_gaussiano_normalizado():
    for tamanho in (3, 5, 7):
        k = F.kernel_gaussiano(tamanho)
        assert np.isclose(k.sum(), 1.0)
        assert k.shape == (tamanho, tamanho)
        assert k[tamanho // 2, tamanho // 2] == k.max()   # centro pesa mais


# ======================================================================
# Filtro da média
# ======================================================================
def test_media_nao_altera_imagem_constante():
    """Uma imagem de valor único deve sair idêntica: a média dos pesos é 1."""
    for valor in (0, 77, 255):
        img = np.full((16, 16), valor, np.uint8)
        for tamanho in (3, 5, 7):
            out = F.filtro_media(img, tamanho)
            assert np.array_equal(out, img), f"valor {valor}, tamanho {tamanho}"


def test_media_reduz_variancia_do_ruido(cinza):
    ruidosa = sal_e_pimenta(cinza, 0.02)
    # A média não é a ferramenta certa para ruído impulsivo, mas deve reduzir
    # a variância local — é o que a suavização faz.
    def variacao_local(x):
        return float(np.abs(np.diff(x.astype(float), axis=1)).mean())

    assert variacao_local(F.filtro_media(ruidosa, 3)) < variacao_local(ruidosa)


# ======================================================================
# Filtro da mediana (critério da seção 13: remove >= 80% do ruído)
# ======================================================================
def test_mediana_remove_ruido_sal_e_pimenta(cinza):
    ruidosa = sal_e_pimenta(cinza, 0.05)
    limpa = F.filtro_mediana(ruidosa, 3)

    erro_antes = np.mean((ruidosa.astype(float) - cinza) ** 2)
    erro_depois = np.mean((limpa.astype(float) - cinza) ** 2)
    assert erro_depois < 0.2 * erro_antes, (
        f"removeu só {100 * (1 - erro_depois / erro_antes):.1f}% do erro"
    )


def test_mediana_remove_ruido_melhor_que_media(cinza):
    """A justificativa teórica do escopo: a mediana preserva bordas, a média não."""
    ruidosa = sal_e_pimenta(cinza, 0.05)

    def erro(x):
        return np.mean((x.astype(float) - cinza) ** 2)

    assert erro(F.filtro_mediana(ruidosa, 3)) < erro(F.filtro_media(ruidosa, 3))


def test_mediana_nao_altera_imagem_constante():
    img = np.full((16, 16), 128, np.uint8)
    assert np.array_equal(F.filtro_mediana(img, 3), img)
    assert np.array_equal(F.filtro_mediana(img, 5), img)


@pytest.mark.skipif(cv2 is None, reason="OpenCV não instalado")
def test_mediana_igual_opencv(cinza):
    ruidosa = sal_e_pimenta(cinza, 0.05)
    for tamanho in (3, 5):
        nossa = F.filtro_mediana(ruidosa, tamanho)
        ref = cv2.medianBlur(ruidosa, tamanho)
        # o interior deve ser idêntico (OpenCV usa padding diferente nas bordas)
        r = tamanho // 2
        assert np.array_equal(nossa[r:-r, r:-r], ref[r:-r, r:-r])


# ======================================================================
# Nitidez
# ======================================================================
def test_nitidez_realca_detalhes(cinza):
    """O realce deve aumentar a energia das transições (sem inverter o sinal)."""
    def energia_bordas(x):
        return float(np.abs(np.diff(x.astype(float), axis=1)).mean())

    for metodo in ("unsharp", "laplaciano"):
        realcada = F.nitidez(cinza, metodo=metodo, intensidade=1.0)
        assert energia_bordas(realcada) > energia_bordas(cinza), metodo


def test_nitidez_intensidade_zero_e_original(cor):
    assert np.array_equal(F.nitidez(cor, metodo="unsharp", intensidade=0), cor)
    assert np.array_equal(F.nitidez(cor, metodo="laplaciano", intensidade=0), cor)


def test_nitidez_maior_intensidade_realca_mais(cinza):
    def energia_bordas(x):
        return float(np.abs(np.diff(x.astype(float), axis=1)).mean())

    fraca = F.nitidez(cinza, intensidade=0.5)
    forte = F.nitidez(cinza, intensidade=2.0)
    assert energia_bordas(forte) > energia_bordas(fraca) > energia_bordas(cinza)


# ======================================================================
# Bordas (Sobel)
# ======================================================================
def test_bordas_em_imagem_uniforme_e_zero():
    """Critério da seção 13: 'sem borda em áreas uniformes'."""
    for valor in (0, 128, 255):
        img = np.full((32, 32), valor, np.uint8)
        assert np.count_nonzero(B.detectar_bordas(img)) == 0


def test_bordas_detectam_degrau_vertical():
    """Uma transição vertical deve produzir resposta grande nas bordas."""
    img = np.zeros((32, 32), np.uint8)
    img[:, 16:] = 255
    gx, gy = B.gradiente_sobel(img)
    assert abs(gx[16, 15]) > 100          # variação na horizontal
    assert np.abs(gy[8:-8, 8:-8]).max() == 0  # nenhuma variação na vertical

    mapa = B.detectar_bordas(img)
    assert mapa[16, 15] > 200             # a borda ficou clara
    assert mapa[16, 2] == 0               # o interior ficou preto


def test_bordas_binarizadas_geram_apenas_0_e_255(cinza):
    for kw in ({"limiar": 80}, {"automatico": True}):
        out = B.detectar_bordas(cinza, binarizar_resultado=True, **kw)
        assert set(np.unique(out)) <= {0, 255}


def test_normalizacao_usa_a_faixa_toda():
    """A normalização deve esticar a magnitude para 0..255 proporcionalmente."""
    magnitude = np.array([[0.0, 5.0], [10.0, 20.0]])
    assert B.normalizar_para_8bits(magnitude).tolist() == [[0, 64], [128, 255]]
    # Sem borda nenhuma (magnitude zero), não pode haver divisão por zero
    assert np.array_equal(B.normalizar_para_8bits(np.zeros((4, 4))), np.zeros((4, 4), np.uint8))


def test_bordas_em_baixo_contraste_ainda_aparecem(cinza):
    """A normalização deve revelar bordas fracas que ficariam invisíveis."""
    fraco = baixo_contraste(cinza)
    mapa = B.detectar_bordas(fraco, normalizar=True)
    assert mapa.max() == 255
    assert np.count_nonzero(mapa > 50) > 0


@pytest.mark.skipif(cv2 is None, reason="OpenCV não instalado")
def test_sobel_igual_opencv(cinza):
    gx, gy = B.gradiente_sobel(cinza)
    ref_gx = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
    ref_gy = cv2.Sobel(cinza, cv2.CV_64F, 0, 1, ksize=3)
    # interior idêntico; as bordas diferem só pelo tipo de padding
    assert np.allclose(gx[1:-1, 1:-1], ref_gx[1:-1, 1:-1])
    assert np.allclose(gy[1:-1, 1:-1], ref_gy[1:-1, 1:-1])


# ======================================================================
# Integração com a melhoria automática (automatico.py procura filtros.py)
# ======================================================================
def test_melhoria_automatica_usa_a_mediana_da_p5(cinza):
    """`automatico._obter_mediana` deve resolver para filtros.filtro_mediana,
    e não para o fallback próprio — é a integração combinada entre P4 e P5."""
    from processamento import automatico as A

    obtida = A._obter_mediana()
    assert obtida is not A._mediana_propria
    assert np.array_equal(obtida(cinza, tamanho=3), F.filtro_mediana(cinza, tamanho=3))


# ======================================================================
# Desempenho (seção 5): filtros < 5 s em 1920x1080
# ======================================================================
@pytest.fixture(scope="module")
def full_hd():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)


@pytest.mark.parametrize("func,params", [
    (F.filtro_media, {"tamanho": 3}),
    (F.filtro_media, {"tamanho": 7}),
    (F.filtro_mediana, {"tamanho": 3}),
    (F.nitidez, {"intensidade": 1.0}),
    (B.detectar_bordas, {}),
])
def test_desempenho_filtros_full_hd(func, params, full_hd):
    t0 = time.perf_counter()
    func(full_hd, **params)
    assert time.perf_counter() - t0 < 5.0
