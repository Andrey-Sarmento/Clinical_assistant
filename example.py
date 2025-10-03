# Carrega função principal
from Python.Predict import avaliar_prontuario

# Carrega um exemplo de prontuário da pasta Dados
with open("Dados/Teste1.txt", encoding="utf-8") as f:
    pront_teste = f.read()

# Primeiros caracteres do prontuário
pront_teste[:300]

# Usa a função principal
saida = avaliar_prontuario(pront_teste)
