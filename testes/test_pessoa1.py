"""
Testes automatizados dos módulos da Pessoa 1 (pytest).
Cobre a seção 12.1 (testes individuais – Núcleo) e os critérios da seção 13
("Desfazer/refazer sem inconsistência em 10 operações seguidas", "abrir arquivo
inválido não derruba o programa").

Executar na raiz do projeto:  python -m pytest testes/ -v
"""

import numpy as np
import pytest
from PIL import Image

from nucleo import ErroImagem, Historico, abrir_imagem, salvar_imagem, sugerir_nome_resultado
from testes.imagens_sinteticas import cena_base, para_cinza


@pytest.fixture
def cinza():
    return para_cinza(cena_base(altura=32, largura=32))


@pytest.fixture
def cor():
    return cena_base(altura=32, largura=32)


def _gravar(caminho, array):
    """Helper: grava um array em disco para os testes de leitura."""
    Image.fromarray(array).save(caminho)
    return caminho


# ======================================================================
# Histórico — seção 12.1 (P1)
# ======================================================================
def test_historico_comeca_no_original(cinza):
    h = Historico(cinza)
    assert h.rotulo_atual == "Original"
    assert np.array_equal(h.imagem_atual, cinza)
    assert np.array_equal(h.imagem_original, cinza)
    assert h.operacoes == []
    assert not h.pode_desfazer()   # botão desabilitado no início (seção 5)
    assert not h.pode_refazer()


def test_empilhar_desfazer_refazer(cinza):
    h = Historico(cinza)
    escura = (cinza * 0.3).astype(np.uint8)
    clara = np.clip(cinza.astype(int) + 40, 0, 255).astype(np.uint8)

    h.empilhar("Escala de cinza", cinza)
    h.empilhar("Brilho", escura)
    h.empilhar("Contraste", clara)

    assert h.operacoes == ["Escala de cinza", "Brilho", "Contraste"]
    assert h.rotulo_atual == "Contraste"
    assert np.array_equal(h.imagem_atual, clara)

    assert np.array_equal(h.desfazer(), escura)
    assert h.rotulo_atual == "Brilho"
    assert h.pode_desfazer() and h.pode_refazer()
    assert h.operacoes == ["Escala de cinza", "Brilho"]

    assert np.array_equal(h.refazer(), clara)
    assert h.rotulo_atual == "Contraste"
    assert not h.pode_refazer()


def test_desfazer_ate_o_original_e_parar(cinza):
    h = Historico(cinza)
    for i in range(3):
        h.empilhar(f"Operação {i + 1}", cinza)

    for _ in range(3):
        assert h.desfazer() is not None
    assert h.rotulo_atual == "Original"
    assert not h.pode_desfazer()
    assert h.desfazer() is None      # desfazer além do original é no-op


def test_restaurar_volta_ao_original_e_limpa_a_pilha(cinza):
    h = Historico(cinza)
    for i in range(4):
        h.empilhar(f"Operação {i + 1}", (cinza + i).astype(np.uint8))

    assert np.array_equal(h.restaurar(), cinza)
    assert h.rotulo_atual == "Original"
    assert h.operacoes == []
    assert h.n_estados == 1
    assert not h.pode_refazer()      # depois de restaurar não há o que refazer


def test_operacao_nova_descarta_o_futuro(cinza):
    """Depois de desfazer, aplicar algo novo sobrescreve o ramo à frente."""
    h = Historico(cinza)
    h.empilhar("A", (cinza + 10).astype(np.uint8))
    h.empilhar("B", (cinza + 20).astype(np.uint8))
    h.desfazer()                      # volta para "A"

    h.empilhar("C", (cinza + 30).astype(np.uint8))
    assert h.operacoes == ["A", "C"]  # "B" foi descartado
    assert not h.pode_refazer()


def test_limite_de_estados_preserva_o_original(cinza):
    """Seção 8 do escopo: limitar a memória do histórico, sem perder o original."""
    h = Historico(cinza, limite=3)
    for i in range(6):
        h.empilhar(f"Operação {i + 1}", (cinza + i).astype(np.uint8))

    assert h.n_estados == 4                                  # Original + 3
    assert h.operacoes == ["Operação 4", "Operação 5", "Operação 6"]
    assert np.array_equal(h.imagem_original, cinza)           # nunca é descartado
    assert np.array_equal(h.restaurar(), cinza)


def test_limite_invalido():
    with pytest.raises(ValueError):
        Historico(np.zeros((4, 4), np.uint8), limite=0)


def test_historico_copia_a_entrada(cinza):
    """Mexer no array de fora não pode alterar o estado guardado."""
    arr = cinza.copy()
    h = Historico(arr)
    arr[:] = 0
    assert np.array_equal(h.imagem_original, cinza)


def test_historico_rejeita_entrada_invalida():
    with pytest.raises(ValueError):
        Historico(None)
    with pytest.raises(ValueError):
        Historico(np.zeros((0, 0), np.uint8))
    with pytest.raises(ValueError):
        Historico(np.zeros((4, 4, 2), np.uint8))


def test_guarda_parametros_para_o_relatorio(cinza):
    h = Historico(cinza)
    h.empilhar("Brilho", cinza, brilho=40)
    assert h.parametros_atuais == {"brilho": 40}
    assert h.estados()[-1].parametros == {"brilho": 40}


def test_empilhar_melhoria_de_um_resultado(cinza):
    """Integração com a P4: aceita o ResultadoMelhoria por duck typing."""
    from processamento import automatico as A
    from testes.imagens_sinteticas import escura

    resultado = A.melhorar_automaticamente(escura(cinza))
    h = Historico(cinza)
    criados = h.empilhar_melhoria(resultado)

    assert len(criados) == len(resultado.passos)
    assert h.operacoes == [p.rotulo for p in resultado.passos]
    assert np.array_equal(h.imagem_atual, resultado.imagem)
    # os parâmetros vieram dos passos, não vazios
    assert h.estados()[-1].parametros == resultado.passos[-1].parametros


def test_dez_operacoes_seguidas_sem_inconsistencia(cinza):
    """Critério da seção 13: desfazer/refazer sem inconsistência em 10 passos."""
    h = Historico(cinza, limite=15)
    imagens = [cinza]
    for i in range(10):
        imagens.append((cinza + i + 1).astype(np.uint8))
        h.empilhar(f"Operação {i + 1}", imagens[-1])

    for i in range(10, 0, -1):
        assert np.array_equal(h.desfazer(), imagens[i - 1])
    for i in range(1, 11):
        assert np.array_equal(h.refazer(), imagens[i])


# ======================================================================
# I/O de imagem — seção 12.1 (P1)
# ======================================================================
def test_abrir_e_salvar_ida_e_volta(tmp_path, cor, cinza):
    for nome, array in (("cena.png", cor), ("cena_cinza.png", cinza),
                        ("cena.bmp", cor), ("cena.jpg", cor)):
        entrada = _gravar(tmp_path / nome, array)
        lida = abrir_imagem(entrada)
        assert lida.dtype == np.uint8
        assert lida.shape == array.shape

        saida = salvar_imagem(lida, tmp_path / "saida" / nome)
        assert saida.exists()
        assert abrir_imagem(saida).shape == array.shape


def test_abrir_devolve_copia_independente(tmp_path, cor):
    caminho = _gravar(tmp_path / "cena.png", cor)
    lida = abrir_imagem(caminho)
    lida[:] = 0
    assert np.any(abrir_imagem(caminho) != 0)


def test_abrir_preserva_alfa(tmp_path, cor):
    alfa = np.full(cor.shape[:2] + (1,), 77, np.uint8)
    rgba = np.concatenate([cor, alfa], axis=2)
    caminho = _gravar(tmp_path / "cena.png", rgba)

    lida = abrir_imagem(caminho)
    assert lida.shape == rgba.shape
    assert np.all(lida[:, :, 3] == 77)


def test_abrir_cinza_fica_com_dois_eixos(tmp_path, cinza):
    caminho = _gravar(tmp_path / "cinza.png", cinza)
    assert abrir_imagem(caminho).ndim == 2


def test_abrir_converte_modos_exoticos(tmp_path, cor):
    """P e 16 bits viram 3 canais / 8 bits em vez de derrubar o programa."""
    paleta = tmp_path / "paleta.png"
    Image.fromarray(cor).convert("P").save(paleta)
    lida = abrir_imagem(paleta)
    assert lida.shape == cor.shape

    dezesseis = tmp_path / "16bits.png"
    Image.fromarray((cor[:, :, 0].astype(np.uint16) * 257)).save(dezesseis)
    lida16 = abrir_imagem(dezesseis)
    assert lida16.dtype == np.uint8
    assert lida16.max() > 200          # escalado, não cortado


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ErroImagem, match="não encontrado"):
        abrir_imagem(tmp_path / "nao_existe.png")


def test_extensao_fora_do_escopo(tmp_path, cor):
    caminho = tmp_path / "cena.gif"
    Image.fromarray(cor).save(caminho)
    with pytest.raises(ErroImagem, match="Formato não suportado"):
        abrir_imagem(caminho)


def test_arquivo_corrompido_nao_derruba(tmp_path):
    """Critério da seção 13: abrir arquivo inválido não derruba o programa."""
    caminho = tmp_path / "falso.png"
    caminho.write_bytes(b"isto definitivamente nao e uma imagem")
    with pytest.raises(ErroImagem):
        abrir_imagem(caminho)


def test_caminho_e_diretorio(tmp_path):
    with pytest.raises(ErroImagem):
        abrir_imagem(tmp_path)


def test_salvar_cria_pastas(tmp_path, cor):
    destino = tmp_path / "a" / "b" / "c" / "resultado.png"
    assert salvar_imagem(cor, destino) == destino
    assert destino.exists()


def test_salvar_jpeg_converte_rgba_para_rgb(tmp_path, cor):
    alfa = np.full(cor.shape[:2] + (1,), 77, np.uint8)
    rgba = np.concatenate([cor, alfa], axis=2)
    destino = salvar_imagem(rgba, tmp_path / "cena.jpg")
    assert abrir_imagem(destino).shape[2] == 3   # JPEG não tem canal alfa


def test_salvar_rejeita_tipo_e_formato_invalidos(tmp_path, cor):
    with pytest.raises(ErroImagem, match="uint8"):
        salvar_imagem(cor.astype(np.float64), tmp_path / "cena.png")   # float sem escala
    with pytest.raises(ErroImagem, match="não suportado"):
        salvar_imagem(cor, tmp_path / "cena.tiff")
    with pytest.raises(ErroImagem):
        salvar_imagem(None, tmp_path / "cena.png")


def test_sugerir_nome_resultado():
    destino = sugerir_nome_resultado("/fotos/lenna.png", "Equalização de histograma")
    assert destino.name == "resultado_equalizacao_de_histograma.png"
    assert destino.parent.name == "resultados"
    assert destino.suffix == ".png"          # mantém a extensão do original

    # acentos, parênteses e espaços não podem sobrar no nome do arquivo
    outro = sugerir_nome_resultado("foto.jpeg", "Detecção de bordas (Sobel)", pasta="saida")
    assert outro.name == "resultado_deteccao_de_bordas_sobel.jpeg"
    assert outro.parent.name == "saida"

    # extensão desconhecida cai para .png em vez de gerar um nome inválido
    assert sugerir_nome_resultado("foto.tiff", "Brilho").suffix == ".png"


def test_sugerir_nome_resultado_so_ascii():
    """Os rótulos da melhoria automática trazem símbolos (γ, ×, →) que não
    podem vazar para o nome do arquivo."""
    from nucleo.imagem_io import _slug

    assert _slug("Correção gama γ=1.87") == "correcao_gama_1_87"
    assert _slug("Filtro da mediana 3×3") == "filtro_da_mediana_3_3"
    assert _slug("Binarização (Otsu)") == "binarizacao_otsu"
    assert _slug("!!!") == "resultado"        # rótulo sem nenhum caractere útil
    for rotulo in ("Correção gama γ=1.87", "Filtro da mediana 3×3", "Nitidez ✨"):
        assert _slug(rotulo).isascii()
