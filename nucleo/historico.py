from dataclasses import dataclass

import numpy as np


@dataclass
class EstadoImagem:
    imagem: np.ndarray
    descricao: str


class HistoricoImagem:
    """
    Controla o histórico de imagens.

    Permite:
    - aplicar uma nova imagem;
    - desfazer;
    - refazer;
    - restaurar a imagem original.
    """

    def __init__(
        self,
        imagem_original: np.ndarray,
        limite: int = 15
    ):
        if not isinstance(imagem_original, np.ndarray):
            raise TypeError(
                "A imagem original deve ser um numpy.ndarray."
            )

        self._original = imagem_original.copy()
        self._atual = imagem_original.copy()

        self._undo: list[EstadoImagem] = []
        self._redo: list[EstadoImagem] = []

        self._descricao_atual = "Original"
        self._limite = limite

    @property
    def original(self) -> np.ndarray:
        """Retorna uma cópia da imagem original."""
        return self._original.copy()

    @property
    def atual(self) -> np.ndarray:
        """Retorna uma cópia da imagem atualmente processada."""
        return self._atual.copy()

    @property
    def descricao_atual(self) -> str:
        """Retorna a descrição do estado atual."""
        return self._descricao_atual

    @property
    def pode_desfazer(self) -> bool:
        """Informa se existe alguma operação para desfazer."""
        return bool(self._undo)

    @property
    def pode_refazer(self) -> bool:
        """Informa se existe alguma operação para refazer."""
        return bool(self._redo)

    def aplicar(
        self,
        nova_imagem: np.ndarray,
        descricao: str
    ) -> None:
        """
        Registra uma nova imagem no histórico.
        """

        if not isinstance(nova_imagem, np.ndarray):
            raise TypeError(
                "A nova imagem deve ser um numpy.ndarray."
            )

        estado_anterior = EstadoImagem(
            imagem=self._atual.copy(),
            descricao=self._descricao_atual
        )

        self._undo.append(estado_anterior)

        # Mantém somente os estados mais recentes
        self._undo = self._undo[-self._limite:]

        self._atual = nova_imagem.copy()
        self._descricao_atual = descricao

        # Uma nova operação elimina o caminho de refazer
        self._redo.clear()

    def desfazer(self) -> np.ndarray:
        """
        Desfaz a última operação, se existir.
        """

        if not self._undo:
            return self.atual

        estado_atual = EstadoImagem(
            imagem=self._atual.copy(),
            descricao=self._descricao_atual
        )

        self._redo.append(estado_atual)

        estado_anterior = self._undo.pop()

        self._atual = estado_anterior.imagem.copy()
        self._descricao_atual = estado_anterior.descricao

        return self.atual

    def refazer(self) -> np.ndarray:
        """
        Refaz uma operação anteriormente desfeita, se existir.
        """

        if not self._redo:
            return self.atual

        estado_atual = EstadoImagem(
            imagem=self._atual.copy(),
            descricao=self._descricao_atual
        )

        self._undo.append(estado_atual)

        proximo_estado = self._redo.pop()

        self._atual = proximo_estado.imagem.copy()
        self._descricao_atual = proximo_estado.descricao

        return self.atual

    def restaurar(self) -> np.ndarray:
        """
        Retorna para a imagem original e limpa o histórico.
        """

        self._undo.clear()
        self._redo.clear()

        self._atual = self._original.copy()
        self._descricao_atual = "Original"

        return self.atual

    def lista_operacoes(self) -> list[str]:
        """
        Retorna as descrições dos estados conhecidos pelo histórico.
        """

        descricoes = [
            estado.descricao
            for estado in self._undo
        ]

        descricoes.append(self._descricao_atual)

        return descricoes
