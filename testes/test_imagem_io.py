import numpy as np
import pytest
from PIL import Image

from nucleo.imagem_io import abrir_imagem, salvar_imagem


def test_abre_e_salva_png(tmp_path):
    origem = tmp_path / "entrada.png"
    destino = tmp_path / "resultados" / "saida.png"

    # Cria uma imagem vermelha de teste
    imagem_original = np.zeros(
        (10, 20, 3),
        dtype=np.uint8
    )
    imagem_original[:, :, 0] = 255

    Image.fromarray(imagem_original).save(origem)

    # Testa a abertura
    imagem_carregada, informacoes = abrir_imagem(origem)

    # Testa o salvamento
    salvar_imagem(imagem_carregada, destino)

    assert imagem_carregada.shape == (10, 20, 3)
    assert imagem_carregada.dtype == np.uint8
    assert informacoes["largura"] == 20
    assert informacoes["altura"] == 10
    assert informacoes["canais"] == 3
    assert destino.exists()


def test_arquivo_invalido_gera_erro(tmp_path):
    arquivo = tmp_path / "arquivo.png"

    # Cria um arquivo que não é realmente uma imagem
    arquivo.write_text("isto não é uma imagem")

    with pytest.raises(ValueError):
        abrir_imagem(arquivo)


def test_extensao_nao_permitida_gera_erro(tmp_path):
    arquivo = tmp_path / "arquivo.gif"
    arquivo.write_bytes(b"qualquer coisa")

    with pytest.raises(ValueError):
        abrir_imagem(arquivo)
