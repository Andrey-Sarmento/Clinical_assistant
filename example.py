# Import the main function
from Python.Predict import avaliar_prontuario

# Load a sample clinical note from the folder "Dados"
with open("Dados/Teste1.txt", encoding="utf-8") as f:
    pront_teste = f.read()

# First characters of the medical record
pront_teste[:300]

# Use the main function
saida = avaliar_prontuario(pront_teste, delta = 30)
