from app.extract_clauses import extract
from app.risk_engine import score

def test_extract_and_risk():
    text = """This Agreement (the “Agreement”) is made and entered into between Alpha Corp and Beta LLC.
Effective Date: January 15, 2024
Payment Terms: Net 60 days from invoice date.
Limit of Liability: USD 15,000,000 aggregate.
Termination: Either party may terminate with 30 days' written notice.
"""
    clauses = extract(text)
    risks = score(clauses, text)
    assert clauses.get("effective_date")
    assert risks["level"] in {"Medium", "High"}
