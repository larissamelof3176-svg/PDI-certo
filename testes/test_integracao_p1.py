import numpy as np
from PIL import Image

from main import AplicacaoPDI


def test_fluxo_completo_do_p1(tmp_path):
    entrada = tmp_path / "entrada.png"
    saida = tmp_path / "resultados" / "saida.png"

    imagem = np.zeros(
        (20, 30, 3),
        dtype=np.uint8
    )

    imagem[:, :, 0] = 100

    Image.fromarray(imagem).save(entrada)

    app = AplicacaoPDI()

    # Abrir
    app.abrir(entrada)

    original = app.historico.atual

    # Simular resultado de uma operação de processamento
    imagem_processada = np.full(
        (20, 30, 3),
        180,
        dtype=np.uint8
    )

    app.aplicar_resultado(
        imagem_processada,
        "Teste de alteração"
    )

    assert np.array_equal(
        app.historico.atual,
        imagem_processada
    )

    # Desfazer
    app.desfazer()

    assert np.array_equal(
        app.historico.atual,
        original
    )

    # Refazer
    app.refazer()

    assert np.array_equal(
        app.historico.atual,
        imagem_processada
    )

    # Salvar
    app.salvar(saida)

    assert saida.exists()
