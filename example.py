# Import the main function
from Python.Predict import evaluate_record

# Load a sample clinical note from the folder "Dados"
with open("Dados/Paciente1.txt", encoding="utf-8") as f:
    pront_teste = f.read()

# First characters of the medical record
pront_teste[:300]

# Use the main function
saida = evaluate_record(pront_teste, delta = 20)
