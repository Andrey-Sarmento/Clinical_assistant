# app.py
import streamlit as st
import streamlit.components.v1 as components
from Python.Predict import highlight_evidence, build_offset_map, normalizar_prontuario, evaluate_record

st.set_page_config(page_title="Análise de Prontuário Médico", layout="wide")

# Estilos customizados do botão
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #3a6ea5;
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 6px;
        font-weight: 500;
    }
    div.stButton > button:first-child:hover {
        background-color: #345e8c;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Forçar fundo claro para toda a página
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;  /* dark suavizado */
        color: #e0e0e0;             /* texto cinza-claro para contraste */
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título e descrição
st.title("Análise de Prontuário Médico")

st.markdown(
    """
    <div style="
        border: 1px solid #444;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #2a2a2a;
    ">
        <p style='font-size:16px; text-align: justify;'>
        O assistente analisa seu prontuário e responde a <b>8 perguntas específicas</b>:
        </p>
        <ol style='font-size:16px; text-align: justify;'>
          <li>O paciente tem doença falciforme?</li>
          <li>O paciente teve internações hospitalares?</li>
          <li>O paciente foi submetido a algum procedimento cirúrgico?</li>
          <li>O paciente recebeu transplante de medula óssea?</li>
          <li>O paciente tomou/usou quelantes de ferro?</li>
          <li>O paciente realizou urinálise com albumina medida?</li>
          <li>O paciente está em terapia transfusional crônica?</li>
          <li>O paciente teve acidente vascular cerebral (infarto)?</li>
        </ol>
        <p style='font-size:16px;'>
        As respostas possíveis são <b>Sim</b>, <b>Não</b> e <b>Sem informação</b>:
        </p>
        <ul style='font-size:16px; text-align: justify;'>
          <li><b>Sim</b> ou <b>Não</b> aparecem apenas quando há evidência explícita positiva ou negativa no texto.</li>
          <li><b>Sem informação</b> significa que não foi encontrada nenhuma menção positiva ou negativa relacionada à pergunta.</li>
        </ul>
        <p style='font-size:16px;'>
        Envie um arquivo <code>.txt</code> ou cole o texto abaixo e clique em <b>Avaliar</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Aviso em destaque
st.warning("⚠️ Este é um protótipo em fase experimental. Os resultados podem conter erros e não devem substituir avaliação médica especializada.")

# Sidebar
with st.sidebar:
    delta = st.number_input("Tamanho da evidência", min_value=0, max_value=200, value=20, step=1)
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
    if st.button("Avaliar", type="primary", use_container_width=True):
        if not texto.strip():
            st.error("Forneça um texto.")
        else:
            with st.spinner("Processando..."):
                df, scores, wei, _ = evaluate_record(texto, delta=delta)

            st.session_state["df0"] = df.copy()
            st.session_state["df"] = df
            st.session_state["texto"] = texto

with col2:
    if st.button("Limpar", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# Execução
if "df0" in st.session_state:
    df0 = st.session_state["df0"]
    df = st.session_state["df"]
    texto = st.session_state["texto"]

    # remove colunas técnicas do dataframe que vai para a tabela e para o CSV
    df = df.drop(columns=["start_norm", "end_norm"], errors="ignore")

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
            {"selector": "td.col3", "props": [("min-width", "640px")]}   # Evidência
        ]
    ).apply(lambda s: [estilo(v, s.name) for v in s], axis=0)

    # renderizar como html e centralizar
    html_table = styled.to_html()
    html_table = f'<div style="display: flex; justify-content: center;">{html_table}</div>'
    st.markdown(html_table, unsafe_allow_html=True)

    # selecionar pergunta para destaque
    # colocar selectbox em uma coluna estreita
    col1, col2 = st.columns([3, 6])  # ajusta proporções
    with col1:
        pergunta_sel = st.selectbox(
            "Escolha uma pergunta:",
            options=["Todas"] + list(df0["pergunta"].unique()),
            index=0
        )

    # destacar evidências no texto
    st.subheader("Prontuário com Evidências Destacadas")

    ttx = normalizar_prontuario(texto, unicode=False)
    texto_norm = normalizar_prontuario(texto, unicode=False)
    mapa = build_offset_map(ttx, texto_norm)
    html_destacado = highlight_evidence(ttx, df0, mapa, pergunta_sel=pergunta_sel)

    # 1. Mostrar a legenda (perguntas + quadradinhos coloridos) em Markdown
    st.markdown(html_destacado.split("</div><br>", 1)[0], unsafe_allow_html=True)

    # 2. Mostrar o texto do prontuário destacado em uma caixa branca centralizada
    prontuario_html = html_destacado.split("</div><br>", 1)[1]

    container_html = f"""
    <div style="
        display: flex;
        justify-content: center;
        margin-top: 10px;">
    <div style="
        background-color: #2a2a2a;
        color: #e0e0e0;
        font-family: Arial, sans-serif;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 0 8px rgba(0,0,0,0.15);
        width: 90%;
        max-width: 1000px;
        text-align: justify;
        line-height: 1.5;
        overflow-y: auto;
        height: 600px;">
        {prontuario_html}
    </div>
    </div>
    """
    components.html(container_html, height=650, scrolling=False)

    # botão de download
    csv = df.to_csv(index=True)
    st.download_button(
        "Baixar resultado (.csv)",
        data=csv,
        file_name="resultado_prontuario.csv",
        mime="text/csv",
        use_container_width=True,
    )
