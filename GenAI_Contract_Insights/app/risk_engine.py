from __future__ import annotations
import re
from typing import Dict, List

RISK_RULES = [
    {
        "id": "liability_high",
        "pattern": r"(limit of liability|liability).*?(\$|USD)?\s*([0-9][0-9,\.]+)",
        "threshold": 10000000,  # $10M
        "message": "Liability appears to exceed $10M; evaluate exposure."
    },
    {
        "id": "payment_term_long",
        "pattern": r"(payment|invoice).*?(net\s*(\d+))",
        "threshold": 45,  # days
        "message": "Payment terms longer than Net 45 may affect cash flow."
    },
    {
        "id": "auto_renewal",
        "pattern": r"(auto\-?renew(al)?|automatic renewal)",
        "message": "Automatic renewal present; ensure notice periods are acceptable."
    },
]

def score(extracted: Dict[str, str], raw_text: str) -> Dict:
    findings: List[Dict] = []
    risk_level = "Low"
    liab = extracted.get("limit_of_liability_amount")
    if liab:
        try:
            v = float(liab.replace(",", "").replace("$", ""))
            if v > 10000000:  # 10M
                findings.append({"rule": "liability_high", "detail": f"Limit: ${v:,.0f}"})
        except Exception:
            pass

    # regex-based rules on raw text
    for rule in RISK_RULES:
        import re as _re
        pat = _re.compile(rule["pattern"], flags=_re.I | _re.S)
        m = pat.search(raw_text)
        if not m:
            continue
        if rule["id"] == "payment_term_long":
            try:
                days = int(m.group(3))
                if days > rule["threshold"]:
                    findings.append({"rule": rule["id"], "detail": f"Net {days}"})
            except Exception:
                pass
        elif rule["id"] == "liability_high":
            try:
                amt = m.group(3).replace(",", "")
                val = float(amt)
                if val > rule["threshold"]:
                    findings.append({"rule": rule["id"], "detail": f"Parsed limit ~ ${val:,.0f}"})
            except Exception:
                pass
        else:
            findings.append({"rule": rule["id"], "detail": "Detected"})

    # Determine overall level
    if any(f["rule"] == "liability_high" for f in findings):
        risk_level = "High"
    elif findings:
        risk_level = "Medium"

    return {"level": risk_level, "findings": findings}
