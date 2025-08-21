import os, time, json, pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from utils_finance import expand_query_finance, extract_numeric_spans, normalize_spaces

class RAGConfig:
    def __init__(self):
        self.DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
        self.INDEX_DIR = self.DATA_DIR / "index"
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # Models
        self.embedding_model = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.cross_encoder_model = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Generation via HF Inference API (handled in ft_client)
        self.gen_repo = os.environ.get("GEN_MODEL_REPO", "google/flan-t5-base")

        # Retrieval settings
        self.max_candidates = int(os.environ.get("MAX_CANDIDATES", 24))
        self.top_k = int(os.environ.get("TOP_K", 5))
        self.max_chunk_chars = int(os.environ.get("MAX_CHUNK_CHARS", 1200))

class SimpleIndex:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.chunks: List[Dict[str, Any]] = []
        self._bm25 = None
        self._dense = None
        self._dense_vectors = None
        self._cross = None

    def _load_texts(self) -> List[str]:
        texts = []
        if (self.cfg.DATA_DIR / "raw").exists():
            for p in sorted((self.cfg.DATA_DIR / "raw").glob("*.txt")):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
        # also allow a single combined file
        if (self.cfg.DATA_DIR / "combined.txt").exists():
            with open(self.cfg.DATA_DIR / "combined.txt","r",encoding="utf-8",errors="ignore") as f:
                texts.append(f.read())
        return texts

    def _chunk_text(self, text: str, max_chars: int) -> List[str]:
        # naive header-aware split
        parts = []
        for para in text.split("\n\n"):
            para = normalize_spaces(para)
            if not para: 
                continue
            if len(para) <= max_chars:
                parts.append(para)
            else:
                # greedy split
                for i in range(0, len(para), max_chars):
                    parts.append(para[i:i+max_chars])
        return parts

    def build(self):
        texts = self._load_texts()
        if not texts:
            raise RuntimeError("No input texts found. Put .txt files under data/raw/ or a combined.txt in data/.")

        # Build chunks
        self.chunks = []
        for doc_id, t in enumerate(texts):
            for ch in self._chunk_text(t, self.cfg.max_chunk_chars):
                self.chunks.append({"doc_id": doc_id, "text": ch})

        # BM25
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

        # Dense
        device = "cuda" if os.environ.get("USE_CUDA","1") == "1" else "cpu"
        self._dense = SentenceTransformer(self.cfg.embedding_model, device=device)
        embs = self._dense.encode([c["text"] for c in self.chunks], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        self._dense_vectors = embs

        # Cross-encoder for reranking (optional; CPU OK, CUDA faster)
        try:
            ce_device = "cuda" if os.environ.get("USE_CUDA","1") == "1" else "cpu"
            self._cross = CrossEncoder(self.cfg.cross_encoder_model, device=ce_device)
        except Exception as e:
            self._cross = None

        # Persist
        with open(self.cfg.INDEX_DIR / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        np.save(self.cfg.INDEX_DIR / "dense.npy", self._dense_vectors)

    def load_or_build(self):
        try:
            with open(self.cfg.INDEX_DIR / "chunks.pkl","rb") as f:
                self.chunks = pickle.load(f)
            self._dense_vectors = np.load(self.cfg.INDEX_DIR / "dense.npy")
        except Exception:
            self.build()
            return

        # rebuild models to query
        device = "cuda" if os.environ.get("USE_CUDA","1") == "1" else "cpu"
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._dense = SentenceTransformer(self.cfg.embedding_model, device=device)
        try:
            ce_device = "cuda" if os.environ.get("USE_CUDA","1") == "1" else "cpu"
            self._cross = CrossEncoder(self.cfg.cross_encoder_model, device=ce_device)
        except Exception:
            self._cross = None

    def retrieve(self, query: str, max_candidates: int = None, top_k: int = None):
        max_candidates = max_candidates or self.cfg.max_candidates
        top_k = top_k or self.cfg.top_k

        # expand query
        qx = expand_query_finance(query)

        # stage 1: BM25 + dense
        bm_scores = self._bm25.get_scores(qx.lower().split())
        bm_scores = (bm_scores - bm_scores.min()) / (bm_scores.ptp() + 1e-9)

        q_vec = self._dense.encode([qx], convert_to_numpy=True, normalize_embeddings=True)[0]
        cs = (self._dense_vectors @ q_vec)  # cosine because normalized
        cs = (cs - cs.min()) / (cs.ptp() + 1e-9)

        # combine
        combo = 0.35*bm_scores + 0.65*cs
        idxs = np.argsort(-combo)[:max_candidates]
        candidates = [{"text": self.chunks[i]["text"], "doc_id": self.chunks[i]["doc_id"], "stage1": float(combo[i])} for i in idxs]

        # stage 2: rerank
        if self._cross is not None:
            pairs = [[query, c["text"][:512]] for c in candidates]
            scores = self._cross.predict(pairs)
            for i, s in enumerate(scores):
                candidates[i]["stage2"] = float(s)
            candidates.sort(key=lambda x: x.get("stage2", x["stage1"]), reverse=True)

        return candidates[:top_k]

def synthesize_answer(question: str, contexts: List[str], ft_client=None) -> Dict[str, Any]:
    start = time.time()
    # If we have a generator client (HF Inference), compose a prompt
    if ft_client is not None:
        prompt = (
            "Answer the financial question using ONLY the provided context. "
            "Quote exact numbers with units and year. If unsure, say 'Insufficient context.'\n\n"
            f"Context:\n- " + "\n- ".join([c[:800] for c in contexts]) + "\n\n"
            f"Question: {question}\nAnswer:"
        )
        try:
            gen = ft_client.generate_text(prompt)
            return {
                "answer": gen.strip(),
                "confidence": 0.78,  # heuristic; could calibrate via cross-enc score
                "response_time": time.time() - start
            }
        except Exception:
            pass

    # Fallback: extractive
    best = contexts[0] if contexts else ""
    return {
        "answer": f"According to the documents: {best[:350]}...",
        "confidence": 0.55,
        "response_time": time.time() - start
    }

class RAGEngine:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.index = SimpleIndex(cfg)

    def startup(self):
        self.index.load_or_build()

    def ask(self, question: str, ft_client=None):
        hits = self.index.retrieve(question)
        contexts = [h["text"] for h in hits]
        ans = synthesize_answer(question, contexts, ft_client=ft_client)
        return {
            "answer": ans["answer"],
            "confidence": ans["confidence"],
            "response_time": ans["response_time"],
            "sources": contexts
        }
