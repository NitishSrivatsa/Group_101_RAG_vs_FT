import re

FIN_TERM_SYNONYMS = {
    "revenue": ["revenue", "net revenue", "total revenue", "sales", "net sales", "total sales"],
    "income": ["net income", "income", "profit", "earnings", "net earnings"],
    "operating income": ["operating income", "operating loss", "income from operations"],
    "ebitda": ["ebitda", "adjusted ebitda"],
    "gross margin": ["gross margin", "gross profit margin"],
    "operating margin": ["operating margin", "opex margin"],
    "capex": ["capex", "capital expenditure", "capital expenditures"],
    "cash": ["cash", "cash & equivalents", "cash and cash equivalents"]
}

def expand_query_finance(q: str) -> str:
    q_low = q.lower()
    expansions = [q]
    for key, syns in FIN_TERM_SYNONYMS.items():
        if key in q_low or any(s in q_low for s in syns):
            expansions += syns
    # de-duplicate preserving order
    seen = set()
    dedup = []
    for tok in expansions:
        if tok not in seen:
            dedup.append(tok); seen.add(tok)
    return " | ".join(dedup)

def extract_numeric_spans(text: str):
    # Very lightweight numeric extraction to support sanity checks
    patt = r"(?:USD|US\$|\$)?\s?[\d\.,]+(?:\s?(?:million|billion|thousand|bn|mn|k|m|b))?"
    return re.findall(patt, text, flags=re.I)

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
