# --------------------------------------------------------------------------------------------
# Bibliotecas
# --------------------------------------------------------------------------------------------

import re
import torch
import unicodedata
import pandas as pd
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------
# Vocabulário e funções de pré-processamento
# --------------------------------------------------------------------------------------------

with open("Dados/voc.txt", encoding="utf-8") as f:
    VOC = f.read().splitlines() # lista de caracteres


def Encoder(text, vocabulary=VOC):
    vocab_index = {char: idx for idx, char in enumerate(vocabulary)}
    return [vocab_index.get(ch, vocab_index["<unk>"]) for ch in text]


def fix_len(seq, target_len, pad=0):
    if len(seq) < target_len:
        return seq + [pad] * (target_len - len(seq))
    return seq[:target_len]


def normalizar_prontuario(x, unicode=True):
    if unicode:
        x = x.lower()  # converte para minúsculas
        x = unicodedata.normalize('NFKD', x)  # separa acentuação
        x = x.encode("ASCII", "ignore").decode("ASCII")  # remove acentos e não ASCII
    
    x = re.sub(r'(\r\n|\r|\n|\\n)+', ' ', x)  # normaliza quebras de linha
    x = re.sub(r'[ \t]+', ' ', x)             # normaliza espaços em branco
    x = x.replace("•", "-")                   # substitui marcador de lista
    x = x.strip()                             # remove espaços em branco nas extremidades
    return x


# --------------------------------------------------------------------------------------------
# Carrega o modelo
# --------------------------------------------------------------------------------------------

device0 = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load("Python/model_traced.pt", map_location=device0)
_ = model.eval()


# --------------------------------------------------------------------------------------------
# Função principal
# --------------------------------------------------------------------------------------------

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


def avaliar_prontuario(pront_teste, delta=30):

    map_pred = {0: "Sim", 1: "Não", 2: "Sem informação"}

    # Prepara entrada
    X = Encoder(normalizar_prontuario(pront_teste, unicode=True))
    X = torch.tensor(fix_len(X, 2**13), dtype=torch.long)
    X = X.unsqueeze(0).to(device0)
    model.eval()
    
    with torch.no_grad():
        logits, scores, wei = model(X)
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
        R = 41   # receptive field
        J = 8    # stride acumulado
        start = max(0, j * J - delta)
        end   = min(2**13, j * J + R + delta)
        frase = normalizar_prontuario(pront_teste, unicode=False)[start:end]
        frase = frase.strip()
        
        linhas.append([pergunta, map_pred[pred_idx], prob_pred, frase])

    # formata como tabela bonita
    col1_w = max(len(l[0]) for l in linhas)
    col2_w = max(len(l[1]) for l in linhas)
    print(f"{'Pergunta'.ljust(col1_w)}  |  {'Pred'.ljust(col2_w)}  |  Prob  |  Evidência")
    print("-" * (col1_w + col2_w + 100))
    for p, pred, prob, frase in linhas:
        print(f"{p.ljust(col1_w)}  |  {pred.ljust(col2_w)}  |  {prob:.3f} |  {frase}")

    return pd.DataFrame(linhas, columns=["pergunta", "pred", "prob", "evidencia"]), scores, wei



# --------------------------------------------------------------------------------------------
# Função para destacar evidências no prontuário
# --------------------------------------------------------------------------------------------

def highlight_evidence(prontuario, df):
    cores = [
        "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e",
        "#9467bd", "#8c564b", "#e377c2", "#6b535b"
    ]

    texto_destacado = prontuario
    legendas = []

    for i, row in df.iterrows():
        if row["pred"] in ["Sim", "Não"]:
            evid = str(row["evidencia"]).strip()
            if evid:
                cor = cores[i % len(cores)]

                # normaliza só para busca
                evid_norm = re.sub(r'\s+', ' ', evid)
                pront_norm = re.sub(r'\s+', ' ', prontuario)

                m = re.search(re.escape(evid_norm), pront_norm)
                if m:
                    # recupera trecho original respeitando quebras
                    start = m.start()
                    end = m.end()
                    trecho_original = prontuario[start:end]

                    span = (
                        f'<span style="border: 2px solid {cor}; '
                        f'padding:2px 4px; border-radius:4px;">{trecho_original}</span>'
                    )
                    texto_destacado = texto_destacado.replace(trecho_original, span, 1)

                    legendas.append(
                        f'<span style="border: 2px solid {cor}; '
                        f'padding:0px 8px; border-radius:3px;"></span> {row["pergunta"]}'
                    )

    legenda_html = "<br>".join(legendas)
    return (
        f"<div>{legenda_html}</div><br>"
        f"<div style='white-space: pre-wrap; text-align: justify;'>{texto_destacado}</div>"
    )


# --------------------------------------------------------------------------------------------
# Exemplo de uso
# --------------------------------------------------------------------------------------------

#with open("Dados/Teste1.txt", encoding="utf-8") as f:
#    pront = f.read()
#    df = avaliar_prontuario(pront)
