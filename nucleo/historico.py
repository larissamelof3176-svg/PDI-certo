"""
nucleo/historico.py
===================
Pilha de estados com desfazer/refazer/restaurar — RF-11 do escopo.
Responsável: Pessoa 1 (P1).

Como funciona
-------------
O histórico guarda uma PILHA DE ESTADOS COMPLETOS: cada item é a imagem
inteira depois de uma operação, e não a diferença entre duas imagens. A
escolha é deliberada:

- Todos os módulos do projeto seguem `funcao(imagem) -> imagem`. Guardar o
  resultado inteiro é a única forma de desfazer que não depende de cada
  operação saber se inverter (o negativo sabe; a equalização, não).
- Desfazer vira mover um índice de posição — O(1), sem recalcular nada.
- O custo é memória. Por isso há um `limite` de estados (seção 8 do escopo:
  "limitar a 10–15 estados"), e as operações pontuais já eram baratas o
  bastante para caber nesse orçamento.

Índice de posição
-----------------
                0          1        2        3
    estados: [Original] [Cinza] [Contraste] [Equalização]
                                        ↑ posicao = 2

    desfazer() → posicao = 1   (volta para "Cinza")
    refazer()  → posicao = 2   (volta para "Contraste")
    restaurar()→ posicao = 0   (volta ao "Original")

Se, depois de desfazer, o usuário aplicar uma operação nova, o "futuro" que
estava à frente é descartado — como em qualquer editor. O estado "Original" é
sempre o índice 0 e NUNCA é removido pelo limite: "Restaurar original"
depende dele.

Uso típico (igual ao descrito no guia da P4):

    historico = Historico(img)
    res = melhorar_automaticamente(img)
    for passo in res.passos:
        historico.empilhar(passo.rotulo, passo.imagem)
    historico.desfazer()          # desfaz o último passo
    historico.imagem_atual        # imagem no ponto atual da pilha
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

LIMITE_PADRAO = 15  # seção 8 do escopo: "limitar a 10–15 estados"


@dataclass
class Estado:
    """Um ponto da pilha: o rótulo, a imagem naquele instante e os parâmetros
    que a produziram (para o relatório explicar o que foi feito)."""
    rotulo: str
    imagem: np.ndarray
    parametros: dict = field(default_factory=dict)


def _validar_array(imagem: np.ndarray) -> np.ndarray:
    """Cópia validada da imagem, para o histórico nunca compartilhar memória
    com quem o chamou (alterar o array de fora não pode mexer no histórico)."""
    if imagem is None:
        raise ValueError("Nenhuma imagem foi fornecida ao histórico.")
    array = np.asarray(imagem)
    if array.size == 0:
        raise ValueError("A imagem está vazia.")
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (1, 3, 4)):
        raise ValueError(
            f"Formato de imagem não suportado: {array.shape}. "
            "Use (altura, largura) ou (altura, largura, 3/4)."
        )
    return np.array(array, copy=True)


class Historico:
    """Pilha de estados de UMA imagem, com desfazer/refazer/restaurar.

    Parâmetros
    ----------
    imagem_original : np.ndarray
        Estado inicial. É copiada na entrada: alterar o array de fora depois
        não afeta o histórico. No sentido inverso não há cópia — as
        propriedades `imagem_atual`/`imagem_original` devolvem o array
        guardado, então quem for MODIFICÁ-LO deve copiar antes.
    limite : int
        Número máximo de OPERAÇÕES guardadas (o "Original" não conta). Ao
        estourar, descarta-se a operação mais antiga — a mais recente é sempre
        a que o usuário pode desfazer.
    """

    def __init__(self, imagem_original: np.ndarray, limite: int = LIMITE_PADRAO):
        if limite < 1:
            raise ValueError("O limite do histórico deve ser pelo menos 1.")
        self._limite = int(limite)
        self._estados: List[Estado] = [Estado("Original", _validar_array(imagem_original))]
        self._posicao = 0

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    @property
    def imagem_original(self) -> np.ndarray:
        """O estado inicial (índice 0) — a âncora do "Restaurar original"."""
        return self._estados[0].imagem

    @property
    def imagem_atual(self) -> np.ndarray:
        """A imagem no ponto atual da pilha (o que a interface deve exibir).

        É o array guardado, não uma cópia (copiar a cada acesso seria caro em
        imagens de 8 MP). Quem precisar modificar deve fazer a cópia.
        """
        return self._estados[self._posicao].imagem

    @property
    def rotulo_atual(self) -> str:
        """Rótulo do estado atual: 'Original' ou o nome da última operação."""
        return self._estados[self._posicao].rotulo

    @property
    def parametros_atuais(self) -> dict:
        """Parâmetros do estado atual (vazio no 'Original')."""
        return self._estados[self._posicao].parametros

    @property
    def operacoes(self) -> List[str]:
        """Rótulos aplicados ATÉ o ponto atual, em ordem cronológica.

        É exatamente a "lista de operações" que o relatório (RF-12) precisa:
        desfazer remove itens daqui, refazer devolve.
        """
        return [e.rotulo for e in self._estados[1:self._posicao + 1]]

    @property
    def posicao(self) -> int:
        return self._posicao

    @property
    def n_estados(self) -> int:
        """Quantidade de estados vivos na pilha (incluindo o 'Original')."""
        return len(self._estados)

    @property
    def limite(self) -> int:
        return self._limite

    def pode_desfazer(self) -> bool:
        """False quando já estamos no 'Original' — a interface deve desabilitar
        o botão nesse caso (requisito de usabilidade, seção 5)."""
        return self._posicao > 0

    def pode_refazer(self) -> bool:
        """False quando não há nada à frente (nada foi desfeito)."""
        return self._posicao < len(self._estados) - 1

    def estados(self) -> Tuple[Estado, ...]:
        """Todos os estados vivos, do mais antigo ao mais recente.

        Inclui o "futuro" de um desfazer, para a interface poder mostrar a
        lista completa de operações com o ponto atual destacado.
        """
        return tuple(self._estados)

    def __iter__(self) -> Iterator[Estado]:
        return iter(self._estados)

    def __len__(self) -> int:
        return len(self._estados)

    def __repr__(self) -> str:
        return (f"Historico({len(self._estados)} estados, posicao={self._posicao}, "
                f"atual='{self.rotulo_atual}')")

    # ------------------------------------------------------------------
    # Modificação
    # ------------------------------------------------------------------
    def empilhar(self, rotulo: str, imagem: np.ndarray, **parametros) -> Estado:
        """Empilha um novo estado e passa a apontá-lo.

        Usado como no guia da P4: `historico.empilhar(passo.rotulo, passo.imagem)`.

        Se houve um `desfazer()` antes, o ramo "refeito" à frente é descartado
        (o trabalho novo substitui o futuro antigo). Ao passar do `limite`,
        a operação mais antiga é descartada — mas nunca o "Original".

        Retorna: o `Estado` criado.
        """
        # Descarta o futuro: aplicar algo novo depois de desfazer sobrescreve
        # o que estava à frente (comportamento padrão de editores).
        del self._estados[self._posicao + 1:]

        estado = Estado(str(rotulo), _validar_array(imagem), dict(parametros))
        self._estados.append(estado)

        # Limite conta só as operações (índices 1..n); o Original fica sempre.
        while len(self._estados) - 1 > self._limite:
            self._estados.pop(1)

        self._posicao = len(self._estados) - 1
        return estado

    def empilhar_melhoria(self, resultado) -> List[Estado]:
        """Empilha todos os passos de um `ResultadoMelhoria` (P4) de uma vez.

        Recebe o objeto por "duck typing" (basta ter `.passos`, e cada passo
        `.rotulo`, `.imagem` e `.parametros`), para o núcleo não precisar
        importar a camada de processamento. Devolve os estados criados.
        """
        criados = []
        for passo in getattr(resultado, "passos", []):
            criados.append(self.empilhar(
                getattr(passo, "rotulo", "Operação"),
                passo.imagem,
                **getattr(passo, "parametros", {}) or {},
            ))
        return criados

    def desfazer(self) -> Optional[np.ndarray]:
        """Volta um passo. Devolve a imagem resultante, ou None se não havia
        o que desfazer (já estamos no 'Original')."""
        if not self.pode_desfazer():
            return None
        self._posicao -= 1
        return self.imagem_atual

    def refazer(self) -> Optional[np.ndarray]:
        """Avança um passo desfeito. Devolve a imagem, ou None se não havia
        nada à frente."""
        if not self.pode_refazer():
            return None
        self._posicao += 1
        return self.imagem_atual

    def restaurar(self) -> np.ndarray:
        """Descarta tudo e volta à imagem original (RF-01: "Restaurar original").

        A pilha de estados é apagada, não só reposicionada: depois de restaurar
        não há o que refazer, o que corresponde à ideia de "recomeçar do zero".
        """
        self._estados = [self._estados[0]]
        self._posicao = 0
        return self.imagem_atual
