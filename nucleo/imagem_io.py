"""
nucleo/imagem_io.py
===================
Leitura e escrita de imagens — RF-01 do escopo.
Responsável: Pessoa 1 (P1).

Responsabilidade
----------------
É a ÚNICA porta de entrada/saída de arquivos do sistema. Converter para cá
todos os `Image.open` espalhados deixa três coisas garantidas num lugar só:

1. Um formato inválido ou arquivo corrompido vira uma `ErroImagem` com
   mensagem em português, pronta para a interface mostrar ao usuário — em vez
   de um traceback do Pillow (requisito de robustez, seção 5 do escopo).
2. Todo array que circula pelo sistema é uint8 com 1, 3 ou 4 canais, que é o
   que os módulos de `processamento/` esperam.
3. Ninguém sobrescreve a imagem original por acidente: `salvar_imagem` sempre
   cria a pasta de destino, e `sugerir_nome_resultado` monta um nome novo.

Formato de retorno
------------------
`abrir_imagem` devolve um np.ndarray uint8:
    (H, W)     → tons de cinza
    (H, W, 3)  → RGB
    (H, W, 4)  → RGBA
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# Formatos dentro do escopo (seção 3): JPG, JPEG, PNG e BMP.
EXTENSOES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp")

# Formatos que o Pillow entende mas que ficam fora do escopo. São aceitos e
# convertidos em vez de recusados: travar a demonstração por causa de um TIFF
# de 16 bits seria pior do que reduzir a profundidade de cor.
PASTA_RESULTADOS_PADRAO = "resultados"


class ErroImagem(Exception):
    """Erro de leitura/escrita de imagem com mensagem amigável ao usuário.

    A interface (P2) deve capturar APENAS esta exceção para exibir a mensagem
    em uma caixa de diálogo — qualquer outra exceção é bug, não erro do usuário.
    """


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def _para_array(img: Image.Image) -> np.ndarray:
    """Converte uma imagem do Pillow em array uint8 de 1, 3 ou 4 canais.

    A maioria dos modos exóticos (paleta, CMYK, tons de cinza com alfa) é
    resolvida pelo próprio Pillow via `convert`. Os modos de 16/32 bits são
    tratados à parte porque `convert("RGB")` simplesmente corta os bits altos,
    o que escureceria a imagem inteira em vez de escalá-la.
    """
    modo = img.mode

    if modo in ("RGB", "RGBA", "L"):
        return np.array(img)

    if modo == "P":
        # Imagem com paleta: só tem transparência se houver a chave 'transparency'
        return np.array(img.convert("RGBA" if "transparency" in img.info else "RGB"))

    if modo == "LA":
        return np.array(img.convert("RGBA"))

    if modo in ("I", "I;16", "I;16B", "I;16L", "F"):
        dados = np.array(img).astype(np.float64)
        maximo = float(dados.max()) if dados.size else 0.0
        if maximo > 255.0:
            dados = dados * (255.0 / maximo)
        return np.clip(np.rint(dados), 0, 255).astype(np.uint8)

    # '1' (preto e branco), 'CMYK' e qualquer outro modo viram RGB
    return np.array(img.convert("RGB"))


def abrir_imagem(caminho) -> np.ndarray:
    """Abre uma imagem do disco e devolve um array uint8.

    Levanta `ErroImagem` (nunca uma exceção do Pillow) quando:
    - o caminho não existe ou não é um arquivo;
    - a extensão está fora de `EXTENSOES_VALIDAS`;
    - o arquivo existe mas não é uma imagem válida / está corrompido.

    O array devolvido é uma CÓPIA independente do arquivo: alterá-lo não mexe
    em nada no disco.
    """
    caminho = Path(caminho)

    if not caminho.exists():
        raise ErroImagem(f"Arquivo não encontrado: {caminho}")
    if not caminho.is_file():
        raise ErroImagem(f"O caminho não aponta para um arquivo: {caminho}")
    if caminho.suffix.lower() not in EXTENSOES_VALIDAS:
        raise ErroImagem(
            f"Formato não suportado: '{caminho.suffix}'. "
            f"Formatos aceitos: {', '.join(EXTENSOES_VALIDAS)}."
        )

    try:
        with Image.open(caminho) as img:
            img.load()  # força a decodificação aqui, para o erro aparecer neste ponto
            array = _para_array(img)
    except UnidentifiedImageError as e:
        raise ErroImagem(
            f"O arquivo '{caminho.name}' não é uma imagem válida ou está corrompido."
        ) from e
    except OSError as e:
        raise ErroImagem(
            f"Não foi possível ler '{caminho.name}': o arquivo pode estar "
            f"truncado ou corrompido ({e})."
        ) from e

    if array.size == 0:
        raise ErroImagem(f"A imagem '{caminho.name}' está vazia.")
    return array


# ---------------------------------------------------------------------------
# Escrita
# ---------------------------------------------------------------------------
def salvar_imagem(imagem: np.ndarray, caminho) -> Path:
    """Grava o array em disco, criando a pasta de destino se necessário.

    Em JPEG (que não tem canal alfa) uma imagem RGBA é convertida para RGB em
    vez de dar erro — o usuário pediu para salvar, não para escolher o formato
    de cor. Nos demais formatos o alfa é preservado.

    Retorna: o caminho efetivamente gravado (como `pathlib.Path`).
    """
    if imagem is None:
        raise ErroImagem("Nenhuma imagem para salvar.")
    array = np.asarray(imagem)
    if array.size == 0:
        raise ErroImagem("A imagem está vazia; nada foi salvo.")
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ErroImagem(
            f"Formato de imagem não suportado para salvar: {array.shape}. "
            "Use (altura, largura) ou (altura, largura, 3/4)."
        )
    if array.dtype != np.uint8:
        # Ninguém espera que salvar mude os valores, então isto é bug do
        # chamador: avisamos em vez de cortar os valores silenciosamente.
        raise ErroImagem(
            f"Esperava uma imagem uint8, recebi {array.dtype}. "
            "Converta com np.clip(...).astype(np.uint8) antes de salvar."
        )

    destino = Path(caminho)
    sufixo = destino.suffix.lower()
    if sufixo not in EXTENSOES_VALIDAS:
        raise ErroImagem(
            f"Formato para salvar não suportado: '{destino.suffix}'. "
            f"Use um de {', '.join(EXTENSOES_VALIDAS)}."
        )

    if sufixo in (".jpg", ".jpeg") and array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]

    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(array).save(destino)
    except (OSError, ValueError) as e:
        raise ErroImagem(f"Não foi possível salvar em '{destino}': {e}") from e
    return destino


# ---------------------------------------------------------------------------
# Nome sugerido para o resultado
# ---------------------------------------------------------------------------
def _slug(texto: str) -> str:
    """Transforma um rótulo legível em um pedaço de nome de arquivo.

    'Equalização de histograma' → 'equalizacao_de_histograma'
    'Correção gama γ=1.87'      → 'correcao_gama_1_87'

    Dois cuidados, ambos para o nome não depender da codificação do sistema de
    arquivos: remove acentos (NFKD + descarte das marcas combinantes) e só
    aceita ASCII — `str.isalnum()` sozinho deixaria passar símbolos como o γ
    dos rótulos gerados pela melhoria automática.
    """
    decomposto = unicodedata.normalize("NFKD", str(texto))
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    limpo = "".join(
        c if (c.isalnum() and c.isascii()) else "_" for c in sem_acento.lower()
    )
    return "_".join(p for p in limpo.split("_") if p) or "resultado"


def sugerir_nome_resultado(caminho_original, operacao: str,
                           pasta: str = PASTA_RESULTADOS_PADRAO) -> Path:
    """Monta um nome de arquivo para o resultado, sem sobrescrever o original.

    Ex.: ('fotos/lenna.png', 'Equalização') → resultados/resultado_equalizacao.png

    Mantém a extensão do original (para não trocar o formato sem avisar).
    """
    original = Path(caminho_original)
    sufixo = original.suffix.lower()
    if sufixo not in EXTENSOES_VALIDAS:
        sufixo = ".png"
    return Path(pasta) / f"resultado_{_slug(operacao)}{sufixo}"
