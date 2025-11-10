# Import the main function
from Python.Predict import evaluate_record

# Load a sample clinical note from the folder "Dados"
with open("Dados/Teste1.txt", encoding="utf-8") as f:
    pront_teste = f.read()

# First characters of the medical record
pront_teste[:300]

# Use the main function
saida = evaluate_record(pront_teste, delta = 20)

# Results with detailed evidence extraction
results = saida[-1]
results["O paciente tem doença falciforme?"]
results["O paciente teve internações hospitalares?"]
results["O paciente foi submetido a algum procedimento cirúrgico?"]
results["O paciente recebeu transplante de medula óssea?"]
results["O paciente tomou/usou quelantes de ferro?"]
results["O paciente realizou urinálise com albumina medida?"]
results["O paciente está em terapia transfusional crônica?"]
results["O paciente teve acidente vascular cerebral (infarto)?"]
