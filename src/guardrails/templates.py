DISCLAIMER = (
    "⚠️ **Disclaimer:** MedRAG-X is for research/education only and is not a medical device. "
    "For diagnosis or treatment decisions, consult a licensed clinician."
)

def safe_answer_wrapper(body: str) -> str:
    return f"{body}\n\n{DISCLAIMER}"