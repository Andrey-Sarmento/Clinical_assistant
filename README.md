# Clinical Assistant

This repository provides a simple interface to run predictions on clinical notes using a pre-trained model.  
The main entry point is the function **`avaliar_prontuario()`**, which takes a clinical record (string) as input and returns a table with the predicted answers to eight standardized clinical questions.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Andrey-Sarmento/Clinical_assistant.git
cd clinical_assistant
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Example of running the main function:

```python
# Import the main function
from Python.predict import avaliar_prontuario

# Load a sample clinical note from the folder "Dados"
with open("Dados/Teste1.txt", encoding="utf-8") as f:
    prontuario = f.read()

# Run the model
df = avaliar_prontuario(prontuario)
```

---

## Output

The output is both:

* Printed in the console in a tabular format
* Returned as a **pandas DataFrame** for further processing.

Example (console print):

```
Pergunta                                                  |  Pred            |  Prob  |  Evidência
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
O paciente tem doença falciforme?                         |  Não             |  1.000 |  s vaso-oclusivas. Não há evidência de doença falciforme. O quadro laboratorial i
O paciente teve internações hospitalares?                 |  Sem informação  |  0.998 |  de infecção local. Durante a internação não houve necessidade de transfusão sa
O paciente foi submetido a algum procedimento cirúrgico?  |  Sim             |  1.000 |  os estáveis. Cirurgia: colecistectomia videolaparoscópica realizada com sucesso.
```

### Explanation of columns:

* **Pergunta**: The clinical question asked about the patient.
* **Pred**: The model’s prediction. One of:

  * `Sim` (Yes)
  * `Não` (No)
  * `Sem informação` (No information in the text)
* **Prob**: Probability of the chosen classification. The closer to **1.0**, the more confident the model is.
* **Evidência**: A fragment of the original clinical note showing the most relevant evidence for that prediction.

---

## Notes

* The model is distributed in TorchScript format (`model_traced.pt`).
* Only `torch` and `pandas` are required beyond Python’s standard library.
* Example clinical notes are provided in the `Dados` folder for testing.

---
