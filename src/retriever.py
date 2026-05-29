from vector_store import query_store


def retrieve(question: str, k: int = 3) -> list[str]:
    """Return the text of the k chunks most relevant to the question."""
    results = query_store(question, k=k)
    return results["documents"][0]      # unwrap the batch nesting, return plain texts


def format_context(chunks: list[str]) -> str:
    """Join retrieved chunks into one labelled context string for the LLM."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Source {i}]\n{chunk}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    question = "How many vacation days do employees get?"

    chunks = retrieve(question, k=3)
    print(f"Retrieved {len(chunks)} chunks\n")

    context = format_context(chunks)
    print("----- context that will be sent to the LLM -----")
    print(context)