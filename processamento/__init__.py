"""
Pacote de processamento do Sistema de Análise, Processamento e Melhoria de
Imagens Digitais (PDI).

Contrato de integração (seção 6 do escopo):
    funcao(imagem: np.ndarray, **parametros) -> np.ndarray

Módulos da Pessoa 4 (P4):
    pontuais     -> cinza, negativo, brilho, contraste, gama, alongamento,
                    binarização (+ Otsu)
    equalizacao  -> equalização de histograma (cinza e cor via luminância)
    automatico   -> diagnóstico e melhoria automática baseada em regras
"""

from .pontuais import (
    escala_cinza,
    negativo,
    ajustar_brilho,
    ajustar_contraste,
    correcao_gama,
    alongamento_contraste,
    limiar_otsu,
    binarizar,
)
from .equalizacao import equalizar_histograma, detalhes_equalizacao
from .automatico import (
    diagnosticar,
    melhorar_automaticamente,
    melhorar,
    LimiaresDiagnostico,
)

__all__ = [
    "escala_cinza",
    "negativo",
    "ajustar_brilho",
    "ajustar_contraste",
    "correcao_gama",
    "alongamento_contraste",
    "limiar_otsu",
    "binarizar",
    "equalizar_histograma",
    "detalhes_equalizacao",
    "diagnosticar",
    "melhorar_automaticamente",
    "melhorar",
    "LimiaresDiagnostico",
]
