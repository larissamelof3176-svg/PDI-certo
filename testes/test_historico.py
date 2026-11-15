import numpy as np

from nucleo.historico import HistoricoImagem


def criar_imagem(valor: int) -> np.ndarray:
    """
    Cria uma imagem pequena para os testes.
    """
    return np.full(
        (3, 3),
        valor,
        dtype=np.uint8
    )


def test_historico_comeca_com_original():
    original = criar_imagem(10)
    historico = HistoricoImagem(original)

    assert np.array_equal(
        historico.atual,
        original
    )

    assert np.array_equal(
        historico.original,
        original
    )

    assert historico.descricao_atual == "Original"
    assert not historico.pode_desfazer
    assert not historico.pode_refazer


def test_desfazer_retorna_estado_anterior():
    historico = HistoricoImagem(criar_imagem(10))

    historico.aplicar(
        criar_imagem(20),
        "Operação 1"
    )

    historico.aplicar(
        criar_imagem(30),
        "Operação 2"
    )

    resultado = historico.desfazer()

    assert np.array_equal(
        resultado,
        criar_imagem(20)
    )

    assert historico.descricao_atual == "Operação 1"


def test_refazer_retorna_estado_desfeito():
    historico = HistoricoImagem(criar_imagem(10))

    historico.aplicar(
        criar_imagem(20),
        "Operação 1"
    )

    historico.desfazer()
    resultado = historico.refazer()

    assert np.array_equal(
        resultado,
        criar_imagem(20)
    )

    assert historico.descricao_atual == "Operação 1"


def test_nova_operacao_apaga_refazer():
    historico = HistoricoImagem(criar_imagem(10))

    historico.aplicar(
        criar_imagem(20),
        "Operação 1"
    )

    historico.desfazer()

    historico.aplicar(
        criar_imagem(40),
        "Nova operação"
    )

    assert not historico.pode_refazer
    assert np.array_equal(
        historico.atual,
        criar_imagem(40)
    )


def test_restaurar_devolve_original():
    original = criar_imagem(10)
    historico = HistoricoImagem(original)

    historico.aplicar(
        criar_imagem(20),
        "Operação 1"
    )

    historico.aplicar(
        criar_imagem(30),
        "Operação 2"
    )

    resultado = historico.restaurar()

    assert np.array_equal(
        resultado,
        original
    )

    assert historico.lista_operacoes() == ["Original"]


def test_historico_nao_compartilha_referencia():
    historico = HistoricoImagem(criar_imagem(10))

    nova_imagem = criar_imagem(20)

    historico.aplicar(
        nova_imagem,
        "Operação"
    )

    # Alteramos a variável original depois de aplicá-la
    nova_imagem[:, :] = 99

    # O histórico não pode ter mudado
    assert np.all(
        historico.atual == 20
    )
