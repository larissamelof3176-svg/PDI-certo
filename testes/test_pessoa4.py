"""
Testes automatizados dos módulos da Pessoa 4 (pytest).
Cobre a seção 12.1 (testes individuais – Pontuais P4) e os critérios da seção 13.

Executar na raiz do projeto:  python -m pytest testes/ -v
"""

import time

import numpy as np
import pytest

from processamento import pontuais as P
from processamento import equalizacao as E
from processamento import automatico as A
from testes.imagens_sinteticas import (
    cena_base, para_cinza, escura, clara, baixo_contraste, sal_e_pimenta, banco_de_teste,
)

try:
    import cv2
except ImportError:  # OpenCV é opcional: só valida
    cv2 = None


@pytest.fixture(scope="module")
def cor():
    return cena_base()


@pytest.fixture(scope="module")
def cinza(cor):
    return para_cinza(cor)


# ======================================================================
# Contrato geral
# ======================================================================
FUNCOES = [
    (P.escala_cinza, {}), (P.negativo, {}), (P.ajustar_brilho, {"brilho": 40}),
    (P.ajustar_contraste, {"alfa": 1.5}), (P.correcao_gama, {"gama": 0.5}),
    (P.alongamento_contraste, {}), (P.binarizar, {"limiar": 100}),
    (P.binarizar, {"automatico": True}), (E.equalizar_histograma, {}), (A.melhorar, {}),
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
    for f in (P.negativo, lambda x: P.ajustar_brilho(x, brilho=30), E.equalizar_histograma):
        out = f(rgba)
        assert out.shape == rgba.shape
        assert np.all(out[:, :, 3] == 77)


def test_entrada_invalida():
    with pytest.raises(ValueError):
        P.negativo(None)
    with pytest.raises(ValueError):
        P.negativo(np.zeros((4, 4, 2), np.uint8))
    with pytest.raises(ValueError):
        P.ajustar_brilho(np.zeros((4, 4), np.uint8), brilho=150)
    with pytest.raises(ValueError):
        P.ajustar_contraste(np.zeros((4, 4), np.uint8), alfa=3)


# ======================================================================
# Pontuais (12.1)
# ======================================================================
def test_escala_cinza_formula():
    img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 150, 200]]], np.uint8)
    esperado = np.rint(np.array([255 * .299, 255 * .587, 255 * .114,
                                 100 * .299 + 150 * .587 + 200 * .114])).astype(np.uint8)
    assert np.array_equal(P.escala_cinza(img)[0], esperado)


def test_negativo_duas_vezes_devolve_original(cor, cinza):
    assert np.array_equal(P.negativo(P.negativo(cor)), cor)
    assert np.array_equal(P.negativo(P.negativo(cinza)), cinza)
    assert np.array_equal(P.negativo(cinza), 255 - cinza)


def test_brilho_respeita_saturacao():
    img = np.array([[0, 10, 128, 240, 255]], np.uint8)
    assert P.ajustar_brilho(img, brilho=50).tolist() == [[50, 60, 178, 255, 255]]
    assert P.ajustar_brilho(img, brilho=-50).tolist() == [[0, 0, 78, 190, 205]]
    assert np.array_equal(P.ajustar_brilho(img, brilho=0), img)


def test_contraste():
    img = np.array([[0, 64, 128, 192, 255]], np.uint8)
    assert np.array_equal(P.ajustar_contraste(img, alfa=1.0), img)
    assert P.ajustar_contraste(img, alfa=2.0).tolist() == [[0, 0, 128, 255, 255]]
    assert np.all(P.ajustar_contraste(img, alfa=0.0) == 128)
    assert P.ajustar_contraste(img, alfa=0.5).tolist() == [[64, 96, 128, 160, 192]]


def test_gama():
    img = np.arange(256, dtype=np.uint8)[None, :]
    assert np.array_equal(P.correcao_gama(img, gama=1.0), img)
    assert P.correcao_gama(img, gama=0.5).mean() > img.mean()
    assert P.correcao_gama(img, gama=2.0).mean() < img.mean()
    g = P.gama_para_alvo(40, 120)
    assert abs(255 * (40 / 255) ** g - 120) < 1


def test_alongamento_ocupa_faixa_toda(cinza):
    bc = baixo_contraste(cinza)
    out = P.alongamento_contraste(bc, 0, 100)
    assert out.min() == 0 and out.max() == 255
    assert out.std() > 2 * bc.std()


def test_binarizacao_gera_so_0_e_255(cor, cinza):
    for img in (cor, cinza):
        for kw in ({"limiar": 90}, {"automatico": True}, {"limiar": 90, "inverter": True}):
            assert set(np.unique(P.binarizar(img, **kw))) <= {0, 255}
    img = np.array([[99, 100, 101]], np.uint8)
    assert P.binarizar(img, limiar=100).tolist() == [[0, 255, 255]]  # f >= T


def test_otsu_bimodal():
    rng = np.random.default_rng(3)
    img = np.concatenate([rng.normal(60, 8, 5000), rng.normal(190, 8, 5000)])
    img = np.clip(img, 0, 255).astype(np.uint8).reshape(100, 100)
    t = P.limiar_otsu(img)
    assert 90 < t < 160


@pytest.mark.skipif(cv2 is None, reason="OpenCV não instalado")
def test_otsu_igual_opencv(cinza):
    for img in (cinza, baixo_contraste(cinza), escura(cinza)):
        t_cv, bin_cv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        assert abs(P.limiar_otsu(img) - t_cv) <= 1
        diff = np.mean(P.binarizar(img, automatico=True) != bin_cv)
        assert diff < 0.01


# ======================================================================
# Equalização (12.1 e 13)
# ======================================================================
def test_equalizacao_espalha_histograma_e_aumenta_desvio(cinza):
    bc = baixo_contraste(cinza)
    eq = E.equalizar_histograma(bc)
    niveis_antes = np.count_nonzero(np.bincount(bc.ravel(), minlength=256))
    assert eq.min() == 0 and eq.max() == 255
    assert eq.std() > bc.std() * 2
    assert np.ptp(eq) > np.ptp(bc)
    assert niveis_antes <= 256


def test_cdf_aproximadamente_linear(cinza):
    d = E.detalhes_equalizacao(baixo_contraste(cinza))
    erro_antes = E.linearidade_cdf(d["cdf_normalizada"])
    erro_depois = E.linearidade_cdf(d["cdf_normalizada_eq"])
    assert erro_depois < erro_antes
    assert erro_depois < 0.06
    assert d["histograma"].sum() == d["n_pixels"]
    assert np.isclose(d["probabilidades"].sum(), 1.0)


def test_formula_do_escopo():
    img = np.array([[52, 55, 61], [59, 79, 61], [76, 61, 52]], np.uint8)
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0]
    esperado = np.rint(255 * (cdf[img] - cdf_min) / (img.size - cdf_min)).astype(np.uint8)
    assert np.array_equal(E.equalizar_histograma(img), esperado)


@pytest.mark.skipif(cv2 is None, reason="OpenCV não instalado")
def test_equalizacao_igual_opencv(cinza):
    for img in (cinza, baixo_contraste(cinza), escura(cinza)):
        diff = np.abs(E.equalizar_histograma(img).astype(int) - cv2.equalizeHist(img).astype(int))
        assert diff.max() <= 1


def test_equalizacao_cor_preserva_matiz(cor):
    bc = baixo_contraste(cor)
    eq_y = E.equalizar_histograma(bc, espaco="ycbcr")
    eq_rgb = E.equalizar_histograma(bc, espaco="rgb")
    eq_hsv = E.equalizar_histograma(bc, espaco="hsv")

    def crominancia(x):
        return E.rgb_para_ycbcr(x)[..., 1:]

    # Equalizar só Y preserva Cb/Cr muito melhor que canal a canal
    err_y = np.abs(crominancia(eq_y) - crominancia(bc)).mean()
    err_rgb = np.abs(crominancia(eq_rgb) - crominancia(bc)).mean()
    assert err_y < err_rgb
    assert P.luminancia(eq_y).std() > P.luminancia(bc).std() * 2
    assert P.luminancia(eq_hsv).std() > P.luminancia(bc).std() * 1.5


def test_conversoes_de_cor_ida_e_volta(cor):
    assert np.abs(E.ycbcr_para_rgb(E.rgb_para_ycbcr(cor)).astype(int) - cor).max() <= 1
    assert np.abs(E.hsv_para_rgb(E.rgb_para_hsv(cor)).astype(int) - cor).max() <= 1


def test_equalizacao_imagem_constante():
    img = np.full((10, 10), 77, np.uint8)
    assert np.array_equal(E.equalizar_histograma(img), img)


# ======================================================================
# Melhoria automática (13: diagnóstico correto em >= 90% do banco)
# ======================================================================
def test_diagnostico_banco_de_teste():
    acertos, total, erros = 0, 0, []
    for semente in range(3):
        for nome, (img, esperado) in banco_de_teste(semente).items():
            d = A.diagnosticar(img)
            total += 1
            if set(d.problemas) == esperado:
                acertos += 1
            else:
                erros.append((nome, semente, d.texto()))
    taxa = acertos / total
    assert taxa >= 0.9, f"taxa {taxa:.0%}; erros: {erros}"


def test_melhoria_imagem_escura(cor):
    res = A.melhorar_automaticamente(escura(cor))
    assert res.alterou
    assert res.passos[0].operacao == "correcao_gama"
    assert 85 <= res.diagnostico_final.brilho <= 170
    assert "muito escura" not in res.diagnostico_final.problemas


def test_melhoria_imagem_clara(cinza):
    res = A.melhorar_automaticamente(clara(cinza))
    assert "muito clara" not in res.diagnostico_final.problemas


def test_melhoria_baixo_contraste(cor):
    bc = baixo_contraste(cor)
    res = A.melhorar_automaticamente(bc)
    assert "baixo contraste" not in res.diagnostico_final.problemas
    assert res.diagnostico_final.desvio > 2 * res.diagnostico_inicial.desvio


def test_melhoria_ruido_aplica_mediana_primeiro(cinza):
    ruidosa = sal_e_pimenta(cinza, 0.05)
    res = A.melhorar_automaticamente(ruidosa)
    assert res.passos[0].operacao == "filtro_mediana"
    erro_antes = np.mean((ruidosa.astype(float) - cinza) ** 2)
    erro_depois = np.mean((res.passos[0].imagem.astype(float) - cinza) ** 2)
    assert erro_depois < 0.2 * erro_antes  # remove >= 80% do erro


def test_imagem_equilibrada_nao_e_alterada(cor):
    res = A.melhorar_automaticamente(cor)
    assert not res.alterou
    assert "Nenhuma melhoria necessária" in res.mensagem
    assert np.array_equal(res.imagem, cor)


def test_passos_reaplicaveis_e_justificados(cor):
    """Cada passo pode ser reaplicado a partir de (operacao, parametros) —
    é o que o histórico precisa para desfazer/refazer."""
    img = baixo_contraste(escura(cor))
    res = A.melhorar_automaticamente(img)
    atual = img
    for p in res.passos:
        atual = A.REGISTRO_OPERACOES[p.operacao](atual, **p.parametros)
        assert np.array_equal(atual, p.imagem)
        assert len(p.justificativa) > 20
    assert np.array_equal(atual, res.imagem)
    assert "Diagnóstico inicial" in res.justificativa()


def test_mediana_propria_igual_opencv(cinza):
    if cv2 is None:
        pytest.skip("OpenCV não instalado")
    r = sal_e_pimenta(cinza)
    ref = cv2.medianBlur(r, 3)
    nossa = A._mediana_propria(r)
    # o interior deve ser idêntico (bordas: OpenCV usa padding diferente)
    assert np.array_equal(nossa[1:-1, 1:-1], ref[1:-1, 1:-1])


# ======================================================================
# Desempenho (seção 5): pontuais < 1 s em 1920x1080
# ======================================================================
@pytest.fixture(scope="module")
def full_hd():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)


@pytest.mark.parametrize("func,params", [
    (P.escala_cinza, {}), (P.negativo, {}), (P.ajustar_brilho, {"brilho": 30}),
    (P.ajustar_contraste, {"alfa": 1.4}), (P.binarizar, {"automatico": True}),
    (E.equalizar_histograma, {}),
])
def test_desempenho_pontuais_full_hd(func, params, full_hd):
    t0 = time.perf_counter()
    func(full_hd, **params)
    assert time.perf_counter() - t0 < 1.0


def test_desempenho_melhoria_full_hd(full_hd):
    t0 = time.perf_counter()
    A.melhorar_automaticamente(full_hd)
    assert time.perf_counter() - t0 < 5.0
