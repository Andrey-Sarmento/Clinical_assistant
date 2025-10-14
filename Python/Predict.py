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

        # trecho normalizado apenas para tabela
        frase_norm = normalizar_prontuario(pront_teste, unicode=False)[start:end].strip()
        
        linhas.append([pergunta, map_pred[pred_idx], prob_pred, frase_norm, start, end])

    df = pd.DataFrame(
        linhas,
        columns=["pergunta", "pred", "prob", "evidencia", "start_norm", "end_norm"]
    )

    return df, scores, wei


# --------------------------------------------------------------------------------------------
# Função para destacar evidências no prontuário
# --------------------------------------------------------------------------------------------

def build_offset_map(original, normalized):
    mapa = []
    i_norm = 0
    for i_orig, c in enumerate(original):
        c_norm = unicodedata.normalize('NFKD', c).encode("ASCII", "ignore").decode("utf-8")
        if not c_norm:
            continue
        for _ in c_norm:
            if i_norm < len(normalized):
                mapa.append(i_orig)
                i_norm += 1
    return mapa


def highlight_evidence(prontuario, df, mapa):
    cores = [
        "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
        "#7f7f7f", "#17becf", "#8c564b", "#bcbd22"
    ]
    color_by_q = {perg: cores[i % len(cores)] for i, perg in enumerate(perguntas)}

    spans = []
    legendas = []

    for _, row in df.iterrows():
        if row["pred"] not in ["Sim", "Não"]:
            continue
        start_norm, end_norm = row["start_norm"], row["end_norm"]
        if pd.isna(start_norm) or pd.isna(end_norm):
            continue

        start_orig = mapa[start_norm]
        end_orig   = mapa[min(end_norm, len(mapa)-1)]

        spans.append((start_orig, end_orig, color_by_q[row["pergunta"]], row["pergunta"]))
        legendas.append(
            f'<span style="display:inline-block; width:14px; height:14px; '
            f'background-color:{color_by_q[row["pergunta"]]}; border-radius:3px; margin-right:6px;"></span>{row["pergunta"]}'
        )

    spans.sort(key=lambda x: x[0])
    saida, pos = [], 0
    for ini, fim, cor, _ in spans:
        if ini > pos:
            saida.append(prontuario[pos:ini])
        saida.append(
            f'<span style="border:2px solid {cor}; padding:2px 4px; border-radius:4px;">'
            f'{prontuario[ini:fim]}</span>'
        )
        pos = fim
    saida.append(prontuario[pos:])

    legenda_html = "<br>".join(legendas)
    return f"<div>{legenda_html}</div><br><div style='white-space: pre-wrap; text-align: justify;'>{''.join(saida)}</div>"


# --------------------------------------------------------------------------------------------
# Exemplo de uso
# --------------------------------------------------------------------------------------------

#with open("Dados/Teste1.txt", encoding="utf-8") as f:
#    pront = f.read()
#    df = avaliar_prontuario(pront)
