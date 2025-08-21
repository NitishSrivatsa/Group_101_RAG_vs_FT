import os
import streamlit as st

from rag_pipeline import RAGConfig, RAGEngine
from ft_client import FTClient

st.set_page_config(page_title="Financial QA: RAG vs Fine-Tuned", layout="wide")
st.title("Financial QA — RAG vs Fine-Tuned (Streamlit)")

with st.sidebar:
    st.header("Configuration")
    use_rag = st.checkbox("Enable RAG", value=True)
    use_ft = st.checkbox("Enable Fine-Tuned (HF Inference)", value=True)
    temperature = st.slider("Temperature (FT)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    st.markdown("**Data directory**: `data/` (add your .txt under `data/raw/`)")
    st.markdown("---")
    st.caption("If you set `HUGGINGFACEHUB_API_TOKEN` and `FT_MODEL_REPO` in Streamlit secrets, the FT model will answer via HF Inference.")

st.info("First run may build an index on CPU — keep data small (e.g., 1–5 annual reports).")

query = st.text_input("Ask a financial question (e.g., 'What was revenue in 2024?')", "")

if "engine" not in st.session_state:
    cfg = RAGConfig()
    eng = RAGEngine(cfg)
    try:
        eng.startup()
    except Exception as e:
        st.warning(f"Index build/load warning: {e}")
    st.session_state.engine = eng

ft = FTClient() if use_ft else None

if st.button("Ask"):
    if not query.strip():
        st.error("Enter a question.")
    else:
        cols = st.columns(2)
        if use_rag:
            with cols[0]:
                st.subheader("RAG")
                res = st.session_state.engine.ask(query, ft_client=ft if (ft and ft.available()) else None)
                st.write(res["answer"])
                st.caption(f"Confidence: {res['confidence']:.2f} • Latency: {res['response_time']:.2f}s")
                with st.expander("Sources"):
                    for i, s in enumerate(res["sources"], 1):
                        st.write(f"**{i}.** {s[:400]}{'...' if len(s)>400 else ''}")
        if use_ft:
            with cols[1]:
                st.subheader("Fine-Tuned (HF Inference)")
                if ft and ft.available():
                    prompt = f"Answer the question concisely with numbers and years.\n\nQuestion: {query}\nAnswer:"
                    try:
                        ans = ft.generate_text(prompt, temperature=temperature)
                    except Exception as e:
                        ans = f"(HF error) {e}"
                else:
                    ans = "No HF token configured. Add HUGGINGFACEHUB_API_TOKEN to Streamlit secrets."
                st.write(ans)
