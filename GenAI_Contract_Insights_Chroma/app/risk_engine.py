from __future__ import annotations
from typing import Dict, List

RISK_RULES = [
    {"id": "liability_high", "threshold": 10000000, "message": "Liability appears to exceed $10M; evaluate exposure."},
    {"id": "payment_term_long", "threshold": 45, "message": "Payment terms longer than Net 45 may affect cash flow."},
    {"id": "auto_renewal", "message": "Automatic renewal present; ensure notice periods are acceptable."},
]

import re

def score(extracted: Dict[str, str], raw_text: str) -> Dict:
    findings: List[Dict] = []
    risk_level = "Low"

    # liability from extracted
    liab = extracted.get("limit_of_liability_amount")
    if liab:
        try:
            v = float(liab.replace(",", "").replace("$", ""))
            if v > 10000000:
                findings.append({"rule": "liability_high", "detail": f"Limit: ${v:,.0f}"})
        except Exception:
            pass

    # payment terms
    m = re.search(r"(payment|invoice).*?(net\s*(\d+))", raw_text, flags=re.I | re.S)
    if m:
        try:
            days = int(m.group(3))
            if days > 45:
                findings.append({"rule": "payment_term_long", "detail": f"Net {days}"})
        except Exception:
            pass

    # auto renewal
    if re.search(r"(auto\-?renew(al)?|automatic renewal)", raw_text, flags=re.I):
        findings.append({"rule": "auto_renewal", "detail": "Detected"})

    if any(f["rule"] == "liability_high" for f in findings):
        risk_level = "High"
    elif findings:
        risk_level = "Medium"

    return {"level": risk_level, "findings": findings}
