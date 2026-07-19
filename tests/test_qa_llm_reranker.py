from common.environments.qa_llm_reranker import parse_query_list, parse_rank_ids


def test_parse_rank_ids_is_bounded_and_deduplicated():
    assert parse_rank_ids("[3, 1, 3, 99]", candidate_count=4, limit=3) == [2, 0]


def test_parse_rank_ids_accepts_plain_text_numbers():
    assert parse_rank_ids("候选 2 和 4", candidate_count=4, limit=2) == [1, 3]


def test_parse_query_list_accepts_json_and_deduplicates():
    assert parse_query_list(
        '说明：["GC-MS 检测规范", "<search>GC-MS 检测规范</search>", "操作 要求"]',
        limit=3,
    ) == ["GC-MS 检测规范", "操作 要求"]


def test_parse_query_list_falls_back_to_numbered_lines():
    assert parse_query_list("1. 缩写 定义\n2、设备 连接方式", limit=2) == [
        "缩写 定义",
        "设备 连接方式",
    ]
