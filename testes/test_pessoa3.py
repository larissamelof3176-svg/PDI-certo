from analise.relatorio import gerar_relatorio
import numpy as np
from analise.metricas import metricas, comparar
from analise.histograma import histograma


def test_imagem_preta():
    m = metricas(np.zeros((10, 10), np.uint8))
    assert m["brilho"] == 0 and m["contraste"] == 0


def test_imagem_branca():
    m = metricas(np.full((10, 10), 255, np.uint8))
    assert m["brilho"] == 255 and m["contraste"] == 0


def test_metade_metade():
    img = np.zeros((10, 10), np.uint8)
    img[:, 5:] = 255
    m = metricas(img)
    assert np.isclose(m["brilho"], img.mean())
    assert np.isclose(m["contraste"], img.std())


def test_soma_histograma_igual_total_de_pixels():
    img = np.random.randint(0, 256, (40, 30, 3), dtype=np.uint8)
    assert histograma(img).sum() == 40 * 30


def test_comparar_delta():
    a = np.full((8, 8), 100, np.uint8)
    b = np.full((8, 8), 150, np.uint8)
    linha = [r for r in comparar(a, b) if r["metrica"] == "brilho"][0]
    assert np.isclose(linha["delta"], 50)


def test_relatorio_gerado(tmp_path):
    a = np.random.randint(0, 80, (60, 80, 3), dtype=np.uint8)
    b = np.clip(a * 2, 0, 255).astype(np.uint8)
    txt, pdf = gerar_relatorio("teste.png", a, b, ["Brilho +40", "Equalizacao"],
                               justificativa="Imagem escura.", pasta=tmp_path)
    conteudo = txt.read_text(encoding="utf-8")
    assert "Brilho +40" in conteudo and "Equalizacao" in conteudo
    assert "brilho" in conteudo
    assert pdf.exists() and pdf.stat().st_size > 0
