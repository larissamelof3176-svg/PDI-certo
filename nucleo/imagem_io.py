from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


# Formatos aceitos pelo projeto
EXTENSOES_PERMITIDAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def abrir_imagem(caminho: str | Path) -> tuple[np.ndarray, dict]:
    """
    Abre uma imagem e devolve:

    1. um array NumPy no formato RGB;
    2. um dicionário com informações do arquivo.
    """

    caminho = Path(caminho)

    # Verifica se o arquivo existe
    if not caminho.exists():
        raise ValueError(f"Arquivo não encontrado: {caminho}")

    # Verifica se a extensão é permitida
    if caminho.suffix.lower() not in EXTENSOES_PERMITIDAS:
        raise ValueError(
            "Formato não permitido. Use arquivos JPG, JPEG, PNG ou BMP."
        )

    try:
        # Abre a imagem usando Pillow
        with Image.open(caminho) as imagem_pillow:
            formato = (
                imagem_pillow.format
                or caminho.suffix.upper().replace(".", "")
            )

            # Força o carregamento completo do arquivo
            imagem_pillow.load()

            # Converte a imagem para RGB
            imagem_pillow = imagem_pillow.convert("RGB")

            # Converte a imagem Pillow para um array NumPy
            imagem = np.asarray(
                imagem_pillow,
                dtype=np.uint8
            ).copy()

    except (UnidentifiedImageError, OSError) as erro:
        raise ValueError(
            "O arquivo está corrompido ou não é uma imagem válida."
        ) from erro

    altura, largura = imagem.shape[:2]

    informacoes = {
        "caminho": str(caminho),
        "nome": caminho.name,
        "formato": formato,
        "largura": int(largura),
        "altura": int(altura),
        "canais": 3,
        "total_pixels": int(largura * altura),
        "tamanho_bytes": int(caminho.stat().st_size),
    }

    return imagem, informacoes


def preparar_para_pillow(imagem: np.ndarray) -> Image.Image:
    """
    Converte um array NumPy para uma imagem que pode ser salva pelo Pillow.

    Aceita:
    - imagem em tons de cinza: formato (altura, largura);
    - imagem RGB: formato (altura, largura, 3).
    """

    imagem = np.asarray(imagem)

    # Garante que os valores estejam entre 0 e 255
    if imagem.dtype != np.uint8:
        imagem = np.clip(imagem, 0, 255).astype(np.uint8)

    # Imagem em tons de cinza
    if imagem.ndim == 2:
        return Image.fromarray(imagem)

    # Imagem colorida RGB
    if imagem.ndim == 3 and imagem.shape[2] == 3:
        return Image.fromarray(imagem)

    raise ValueError(
        "A imagem deve ser 2D em tons de cinza ou RGB com três canais."
    )


def salvar_imagem(
    imagem: np.ndarray,
    caminho: str | Path
) -> Path:
    """
    Salva uma imagem no caminho informado.

    A pasta de destino é criada automaticamente se não existir.
    """

    caminho = Path(caminho)

    # Cria a pasta de destino, se necessário
    caminho.parent.mkdir(parents=True, exist_ok=True)

    # Converte e salva usando Pillow
    imagem_pillow = preparar_para_pillow(imagem)
    imagem_pillow.save(caminho)

    return caminho




