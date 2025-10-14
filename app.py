# app.py
import streamlit as st
from Python.Predict import avaliar_prontuario

st.set_page_config(page_title="Assistente de Diagnóstico", layout="wide")

st.title("Assistente de Diagnóstico")
st.caption("Envie um prontuário no formato .txt ou cole o texto abaixo para avaliação.")

# Sidebar
with st.sidebar:
    delta = st.number_input("delta", min_value=0, max_value=200, value=30, step=1)
    st.markdown("---")
    st.caption("Use UTF-8 nos arquivos .txt.")

# Tabs
tab1, tab2 = st.tabs(["Upload .txt", "Colar texto"])
texto = ""

with tab1:
    up = st.file_uploader("Carregue um prontuário (.txt)", type=["txt"])
    if up is not None:
        texto = up.read().decode("utf-8")

with tab2:
    texto = st.text_area("Cole o prontuário aqui", height=240, value=texto)

# Botões
col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("Avaliar", type="primary", use_container_width=True)
with col2:
    clear = st.button("Limpar", use_container_width=True)

if clear:
    st.experimental_rerun()

# Execução
if run:
    if not texto.strip():
        st.error("Forneça um texto.")
    else:
        with st.spinner("Processando..."):
            df, scores, wei = avaliar_prontuario(texto, delta=delta)

        # ajustar índice para começar em 1
        df.index = range(1, len(df) + 1)
        df = df.rename(columns={
            "pergunta": "Pergunta",
            "pred": "Predição",
            "prob": "Probabilidade",
            "evidencia": "Evidência no Prontuário"
        })
        df["Probabilidade"] = df["Probabilidade"].map(lambda x: f"{x:.3f}")

        st.subheader("Resultado")
        # aplicar estilos por coluna
        def estilo(val, col):
            if col == "Predição" or col == "Probabilidade":
                return 'text-align: center; padding: 6px;'
            elif col == "Evidência no Prontuário":
                return 'text-align: justify; padding: 6px; max-width: 500px; white-space: normal; word-wrap: break-word;'
            else:  # Pergunta
                return 'text-align: left; padding: 6px;'

        styled = df.style.set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center"), ("padding", "8px")]},
                {"selector": "td", "props": [("padding", "8px")]},
                {"selector": "td.col0", "props": [("min-width", "340px")]},  # Pergunta
                {"selector": "td.col1", "props": [("min-width", "140px")]},  # Predição
                {"selector": "td.col2", "props": [("min-width", "140px")]},  # Probabilidade
                {"selector": "td.col3", "props": [("min-width", "540px")]}   # Evidência
            ]
        ).apply(lambda s: [estilo(v, s.name) for v in s], axis=0)

        # renderizar como html e centralizar
        html_table = styled.to_html()
        html_table = f'<div style="display: flex; justify-content: center;">{html_table}</div>'

        st.markdown(html_table, unsafe_allow_html=True)


        # botão de download
        csv = df.to_csv(index=True)
        st.download_button(
            "Baixar resultado (.csv)",
            data=csv,
            file_name="resultado_prontuario.csv",
            mime="text/csv",
            use_container_width=True,
        )
