import sys
import numpy as np
from PIL import Image
from processamento import automatico as A
from analise.relatorio import gerar_relatorio

caminho = sys.argv[1]
img = np.array(Image.open(caminho).convert("RGB"))
res = A.melhorar_automaticamente(img)
ops = [p.rotulo for p in res.passos]
print(gerar_relatorio(caminho, img, res.imagem, ops, res.justificativa(), caminho))