from __future__ import annotations
import re
from typing import Dict

CLAUSE_PATTERNS = {
    "parties": r"(between\s+)([A-Za-z0-9 &.,'-]+)(\s+and\s+)([A-Za-z0-9 &.,'-]+)",
    "effective_date": r"(effective\s+date[:\s]*)([A-Za-z]+\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})",
    "termination": r"(termination[^\n]*?:?\s*)([\s\S]{0,240})",
    "payment_terms": r"(payment terms?|invoice terms?)[:\s]*([\s\S]{0,160})",
    "limit_of_liability": r"(limit of liability|liability cap)[:\s]*([\s\S]{0,160})",
}

AMOUNT_PATTERN = r"(\$|USD)\s*([0-9][0-9,\.]+)"
NET_PATTERN = r"net\s*(\d+)"

def extract(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, pat in CLAUSE_PATTERNS.items():
        m = re.search(pat, text, flags=re.I)
        if not m: 
            continue
        val = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(0)
        out[key] = val.strip()

    m_amt = re.search(AMOUNT_PATTERN, out.get("limit_of_liability", ""), flags=re.I)
    if m_amt:
        out["limit_of_liability_amount"] = m_amt.group(2)

    m_net = re.search(NET_PATTERN, out.get("payment_terms", ""), flags=re.I)
    if m_net:
        out["payment_terms_net_days"] = m_net.group(1)

    return out
