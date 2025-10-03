# predict.py
import torch
import torch.nn.functional as F
import pandas as pd
import re, unicodedata

# --------------------------------------------------
# Carregar vocabulário
# --------------------------------------------------
with open("Dados/voc.txt", encoding="utf-8") as f:
    VOC = f.read().splitlines()

def Encoder(text, vocabulary=VOC):
    vocab_index = {char: idx for idx, char in enumerate(vocabulary)}
    return [vocab_index.get(ch, vocab_index["<unk>"]) for ch in text]


def fix_len(seq, target_len, pad=0):
    if len(seq) < target_len:
        return seq + [pad] * (target_len - len(seq))
    return seq[:target_len]


def normalizar_prontuario(x):
    x = x.lower()
    x = unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8')
    x = re.sub(r'(\r\n|\r|\n|\\n)+', '\n', x)
    x = re.sub(r'[ \t]+', ' ', x)
    x = re.sub(r'\n+', '\n', x)
    return x.strip()



# --------------------------------------------------
# Carregar modelo traçado
# --------------------------------------------------
device0 = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load("Python/model_traced.pt", map_location=device0)
_ = model.eval()

# --------------------------------------------------
# Função de avaliação
# --------------------------------------------------
def avaliar_prontuario(pront_teste, delta=20):

    perguntas = [
        "O paciente tem doença falciforme?",
        "O paciente teve internações hospitalares?",
        "O paciente foi submetido a algum procedimento cirúrgico?",
        "O paciente recebeu transplante de medula óssea?",
        "O paciente tomou/usou quelantes de ferro?",
        "O paciente realizou urinálise com albumina medida?",
        "O paciente está em terapia transfusional crônica?",
        "O paciente teve acidente vascular cerebral (infarto)?"
    ]

    map_pred = {0: "Sim", 1: "Não", 2: "Sem informação"}

    # Prepara entrada
    X = Encoder(normalizar_prontuario(pront_teste))
    X = torch.tensor(fix_len(X, 8192), dtype=torch.long).unsqueeze(0).to(device0)

    with torch.no_grad():
        logits, scores, _ = model(X)              # modelo retorna (logits, scores)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        preds = preds.squeeze(0)
        probs = probs.squeeze(0).T  # (8,3)

    linhas = []
    for i, pergunta in enumerate(perguntas):
        pred_idx = preds[i].item()
        prob_pred = round(float(probs[i][pred_idx]), 3)

        # encontra token mais relevante para essa pergunta
        j = torch.argmax(scores[0, :, i]).item()

        # extrai janela de texto em torno do token j
        start = max(0, 2*(2*(2*j+1)+1)+1 - delta)
        end   = min(len(pront_teste), 2*(2*(2*j+7)+7)+5 + delta)
        frase = pront_teste[start:end]
        frase = frase.replace("\n", " ")       # remove quebras de linha
        frase = " ".join(frase.split())        # normaliza espaços em branco

        linhas.append([pergunta, map_pred[pred_idx], prob_pred, frase])

    # formata como tabela bonita
    col1_w = max(len(l[0]) for l in linhas)
    col2_w = max(len(l[1]) for l in linhas)
    print(f"{'Pergunta'.ljust(col1_w)}  |  {'Pred'.ljust(col2_w)}  |  Prob  |  Evidência")
    print("-" * (col1_w + col2_w + 100))
    for p, pred, prob, frase in linhas:
        print(f"{p.ljust(col1_w)}  |  {pred.ljust(col2_w)}  |  {prob:.3f} |  {frase}")

    # também retorna em DataFrame se quiser manipular
    return pd.DataFrame(linhas, columns=["pergunta", "pred", "prob", "evidencia"])

# --------------------------------------------------
# Exemplo de uso
# --------------------------------------------------
#with open("Dados/Teste1.txt", encoding="utf-8") as f:
#    pront = f.read()
#    df = avaliar_prontuario(pront)
