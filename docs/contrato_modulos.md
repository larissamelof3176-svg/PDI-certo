# Contrato entre os módulos

## Representação das imagens

As imagens serão representadas por arrays do NumPy.

- Imagem RGB: formato `(altura, largura, 3)`.
- Imagem em tons de cinza: formato `(altura, largura)`.
- Tipo padrão: `uint8`.
- Intensidades entre 0 e 255.
- As funções não devem alterar diretamente a imagem recebida.
- Cada função deve retornar uma nova imagem.

## Módulos de processamento

As funções de processamento devem seguir o padrão:

```python
funcao(imagem, **parametros) -> np.ndarray
