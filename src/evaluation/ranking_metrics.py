def mean_reciprocal_rank(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0
