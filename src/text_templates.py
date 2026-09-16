"""Single source of truth for how a query and a misconception become text.

Every stage -- baselines, bi-encoder, reranker, distillation, serving -- imports
these. If the template drifts between training and serving the whole thing is
silently broken, so it lives in exactly one place.
"""

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def query_text(row) -> str:
    """row: mapping with subject/construct/question/correct/incorrect."""
    return (
        f"Subject: {row['subject']} | Topic: {row['construct']}\n"
        f"Question: {row['question']}\n"
        f"Correct answer: {row['correct']}\n"
        f"Student chose: {row['incorrect']}"
    )


def doc_text(misconception_name: str) -> str:
    return f"Misconception: {misconception_name}"


def pair_text(row, misconception_name: str):
    """Cross-encoder input pair."""
    return query_text(row), doc_text(misconception_name)
