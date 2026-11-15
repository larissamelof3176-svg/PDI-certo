from nucleo.historico import HistoricoImagem
from nucleo.imagem_io import abrir_imagem, salvar_imagem


class AplicacaoPDI:
    """
    Classe principal responsável por coordenar
    abertura, processamento, histórico e salvamento.
    """

    def __init__(self):
        self.imagem_original = None
        self.informacoes_arquivo = None
        self.historico = None

    def abrir(self, caminho):
        """
        Abre uma imagem e cria o histórico inicial.
        """

        imagem, informacoes = abrir_imagem(caminho)

        self.imagem_original = imagem.copy()
        self.informacoes_arquivo = informacoes
        self.historico = HistoricoImagem(imagem)

        return imagem

    def aplicar_resultado(
        self,
        nova_imagem,
        descricao
    ):
        """
        Registra no histórico o resultado de uma operação.
        """

        if self.historico is None:
            raise ValueError(
                "Nenhuma imagem foi carregada."
            )

        self.historico.aplicar(
            nova_imagem,
            descricao
        )

    def salvar(self, caminho):
        """
        Salva a imagem processada atual.
        """

        if self.historico is None:
            raise ValueError(
                "Nenhuma imagem foi carregada."
            )

        return salvar_imagem(
            self.historico.atual,
            caminho
        )

    def desfazer(self):
        """
        Desfaz a última operação.
        """

        if self.historico is None:
            raise ValueError(
                "Nenhuma imagem foi carregada."
            )

        return self.historico.desfazer()

    def refazer(self):
        """
        Refaz a última operação desfeita.
        """

        if self.historico is None:
            raise ValueError(
                "Nenhuma imagem foi carregada."
            )

        return self.historico.refazer()

    def restaurar(self):
        """
        Retorna para a imagem original.
        """

        if self.historico is None:
            raise ValueError(
                "Nenhuma imagem foi carregada."
            )

        return self.historico.restaurar()


if __name__ == "__main__":
    print("Núcleo P1 carregado com sucesso.")
