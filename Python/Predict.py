# --------------------------------------------------------------------------------------------
# Bibliotecas
# --------------------------------------------------------------------------------------------

import re
import torch
import unicodedata
import pandas as pd
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------
# Funções de pré-processamento
# --------------------------------------------------------------------------------------------

with open("Dados/voc.txt", encoding="utf-8") as f:
    VOC = f.read().splitlines() # lista de caracteres

vocab_index = {char: idx for idx, char in enumerate(VOC)}


def Encoder(text):
    """Converte texto em lista de índices do vocabulário."""
    return [vocab_index.get(ch, vocab_index["<unk>"]) for ch in text]


def fix_len(seq, target_len, pad=0):
    """Trunca ou preenche a sequência até target_len."""
    n = len(seq)
    if n >= target_len:
        return seq[:target_len]
    out = [pad] * target_len
    out[:n] = seq
    return out


def normalizar_prontuario(x, unicode=True):
    """Normaliza texto: minúsculas, acentos, espaços e marcadores."""
    if unicode:
        x = x.lower()
        x = unicodedata.normalize('NFKD', x)
        x = x.encode("ASCII", "ignore").decode("ASCII")

    x = re.sub(r'(\r\n|\r|\n|\\n)+', ' ', x)
    x = re.sub(r'[ \t]+', ' ', x)
    x = x.replace("•", "-").strip()
    return x


# --------------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------------

device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("Python/model_traced.pt", map_location=device0)
_ = model.eval()  # modo avaliação (desativa dropout, batchnorm)


def extract_evidences(
    scores, i, texto_norm, pred,
    delta=20, k=3, R=41, J=8,
    min_ratio=0.2, sum_threshold=0.8, min_distance=5
):
    """
    Extrai evidências do texto normalizado para a questão i.
    Retorna lista de (start, end, trecho, peso).
    """
    n = len(texto_norm)
    s = scores[0, :, i].cpu().numpy()
    idxs = s.argsort()[::-1]

    evidences = []
    total_weight = s[idxs[:k]].sum()

    for j in idxs:
        center = j * J
        if center >= n:
            continue
        if any(abs(center - (j2 * J)) < (min_distance * J) for _, _, _, _, j2 in evidences):
            continue

        start = max(0, center - delta)
        end   = min(n, center + R + delta)
        trecho = texto_norm[start:end].strip()
        if not trecho:
            continue

        evidences.append((start, end, trecho, float(s[j]), j))
        if len(evidences) >= k:
            break

    if not evidences:
        # fallback: pega o primeiro válido
        for j in idxs:
            center = j * J
            if center < n:
                start = max(0, center - delta)
                end   = min(n, center + R + delta)
                trecho = texto_norm[start:end].strip()
                evidences = [(start, end, trecho, float(s[j]), j)]
                break

    # filtros
    top1 = evidences[0][3]
    evidences = [ev for ev in evidences if ev[3] >= min_ratio * top1]

    if total_weight >= sum_threshold:
        evidences = evidences[:k]

    if pred == "Sem informação":
        evidences = evidences[:1]

    return [(s0, e0, t0, w0) for s0, e0, t0, w0, _ in evidences]


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
MAP_PRED = {0: "Sim", 1: "Não", 2: "Sem informação"}


def evaluate_record(pront_teste, delta=20):
    texto_norm = normalizar_prontuario(pront_teste, unicode=True)
    X = Encoder(texto_norm)
    X = torch.tensor(fix_len(X, 2**13), dtype=torch.long).unsqueeze(0).to(device0)
    
    with torch.no_grad():
        logits, scores, wei = model(X)
        probs = F.softmax(logits, dim=1).squeeze(0).T  # (8,3)
        preds = torch.argmax(logits, dim=1).squeeze(0)

    mapa = build_offset_map(pront_teste, texto_norm)
    linhas = []

    for i, pergunta in enumerate(perguntas):
        pred_idx = preds[i].item()
        pred_label = MAP_PRED[pred_idx]
        prob_pred = round(float(probs[i][pred_idx]), 3)

        evids = extract_evidences(scores, i, texto_norm, pred_label, delta=delta)

        trechos = [
            normalizar_prontuario(pront_teste, unicode=False)[mapa[s]:mapa[min(e, len(mapa)-1)]].strip()
            for s, e, _, _ in evids
        ]
        evidencias_texto = "• " + "<br>• ".join(trechos) if trechos else ""

        starts, ends = zip(*[(s, e) for s, e, _, _ in evids]) if evids else ([], [])

        linhas.append([pergunta, pred_label, prob_pred, evidencias_texto, list(starts), list(ends)])

    df = pd.DataFrame(
        linhas,
        columns=["pergunta", "pred", "prob", "evidencia", "start_norm", "end_norm"]
    )
    df["evidencia"] = df["evidencia"].str.replace(r"\s*\n\s*", " ", regex=True)

    return df, scores, wei, mapa


# --------------------------------------------------------------------------------------------
# Função para destacar evidências no prontuário
# --------------------------------------------------------------------------------------------

def build_offset_map(original: str, normalized: str):
    """
    Cria mapa de índices: cada posição do texto normalizado aponta
    para o índice correspondente no texto original.
    Garante consistência tratando acentos, espaços, tabs e quebras de linha.
    """
    mapa = []
    i_norm = 0

    for i_orig, c in enumerate(original):
        # normaliza acento -> ASCII
        c_norm = unicodedata.normalize("NFKD", c).encode("ASCII", "ignore").decode("utf-8")

        # espaços, tabs e quebras viram " "
        if c in "\r\n\t":
            c_norm = " "

        # normaliza bullets
        if c == "•":
            c_norm = "-"

        # se depois de tudo não sobrou nada, ignora
        if not c_norm:
            continue

        # percorre cada caractere normalizado
        for _ in c_norm:
            if i_norm >= len(normalized):
                return mapa
            mapa.append(i_orig)
            i_norm += 1

    return mapa


def highlight_evidence(prontuario, df, mapa, pergunta_sel="Todas"):
    cores = [
        "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
        "#7f7f7f", "#17becf", "#8c564b", "#bcbd22"
    ]
    color_by_q = {perg: cores[i % len(cores)] for i, perg in enumerate(perguntas)}

    spans, legendas = [], []
    for _, row in df.iterrows():
        if row["pred"] not in ["Sim", "Não"]:
            continue
        if pergunta_sel != "Todas" and row["pergunta"] != pergunta_sel:
            continue

        starts, ends = row["start_norm"], row["end_norm"]
        if not isinstance(starts, list):
            starts, ends = [starts], [ends]

        for s, e in zip(starts, ends):
            if pd.isna(s) or pd.isna(e):
                continue
            s, e = int(s), int(e)
            if s < 0 or e <= s or s >= len(mapa):
                continue
            start_orig = mapa[s]
            end_orig   = mapa[min(e, len(mapa) - 1)]
            spans.append((start_orig, end_orig, color_by_q[row["pergunta"]]))

        legendas.append(
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background-color:{color_by_q[row["pergunta"]]};border-radius:3px;'
            f'margin-right:6px;"></span>{row["pergunta"]}'
        )

    spans.sort(key=lambda x: x[0])

    merged = []
    for ini, fim, cor in spans:
        if merged and cor == merged[-1][2] and ini <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], fim), cor)
        else:
            merged.append((ini, fim, cor))

    saida, pos = [], 0
    for ini, fim, cor in merged:
        if ini > pos:
            saida.append(prontuario[pos:ini])
        saida.append(
            f'<span style="border:2px solid {cor};padding:2px 4px;border-radius:4px;">'
            f'{prontuario[ini:fim]}</span>'
        )
        pos = fim
    if pos < len(prontuario):
        saida.append(prontuario[pos:])

    legenda_html = "<br>".join(legendas)
    corpo_html = "".join(saida)

    return (
        f"<div>{legenda_html}</div><br>"
        f"<div style='white-space:pre-wrap;text-align:justify;'>{corpo_html}</div>"
    )
