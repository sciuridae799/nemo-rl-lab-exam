from common.environments.qa_llm_reranker import parse_rank_ids


def test_parse_rank_ids_is_bounded_and_deduplicated():
    assert parse_rank_ids("[3, 1, 3, 99]", candidate_count=4, limit=3) == [2, 0]


def test_parse_rank_ids_accepts_plain_text_numbers():
    assert parse_rank_ids("候选 2 和 4", candidate_count=4, limit=2) == [1, 3]
