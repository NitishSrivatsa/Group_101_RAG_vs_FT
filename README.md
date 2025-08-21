# Financial QA — RAG vs Fine-Tuned (Streamlit)

This app compares a Retrieval-Augmented Generation (RAG) pipeline with a Fine-Tuned (seq2seq) model for financial questions.

## Quick Start (Streamlit Community Cloud)

1. **Prepare a repo** with these files.
2. Add small input docs as plain text under `data/raw/` (e.g., copy/paste sections from PDFs).
3. In Streamlit Cloud:
   - Create a new app from your GitHub repo.
   - (Optional) In **Secrets**, add:
     - `HUGGINGFACEHUB_API_TOKEN` = your token
     - `FT_MODEL_REPO` = your fine-tuned model id (e.g., `your-username/finqa-flan-t5`)
   - Deploy with `app.py` as the entrypoint.
4. First run will build the index on CPU. Keep data modest (1–5 reports).

## Local Dev

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Colab → HF Hub (Fine-Tune & Host)

- Fine-tune a small seq2seq model (e.g., `google/flan-t5-base`) on your Q/A.
- Push to Hugging Face Hub and use Inference API from this app.

## Data Layout

```
data/
  raw/
    2023_report.txt
    2024_report.txt
  index/
    (auto-created: chunks.pkl, dense.npy)
```
