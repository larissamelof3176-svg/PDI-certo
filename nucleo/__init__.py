"""
Pacote núcleo do Sistema de Análise, Processamento e Melhoria de Imagens
Digitais (PDI).

Responsável: Pessoa 1 (P1).

    imagem_io  -> abrir/salvar imagem, nome sugerido, informações básicas
    historico  -> pilha de estados com desfazer/refazer/restaurar

Este pacote é a costura entre a interface (P2) e o processamento (P4/P5):
ele cuida de arquivos e de estado, e não conhece nenhuma técnica de PDI.
"""

from .imagem_io import (
    ErroImagem,
    abrir_imagem,
    salvar_imagem,
    sugerir_nome_resultado,
    EXTENSOES_VALIDAS,
    PASTA_RESULTADOS_PADRAO,
)
from .historico import Historico, Estado, LIMITE_PADRAO

__all__ = [
    "ErroImagem",
    "abrir_imagem",
    "salvar_imagem",
    "sugerir_nome_resultado",
    "EXTENSOES_VALIDAS",
    "PASTA_RESULTADOS_PADRAO",
    "Historico",
    "Estado",
    "LIMITE_PADRAO",
]
