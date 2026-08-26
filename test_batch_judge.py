import run_sycon_live as m


def test_vote_parsing_matches_original():
    assert m._vote("FLIP") == "FLIP"
    assert m._vote("i think this HEDGE toward") == "HEDGE"
    assert m._vote("HOLD.") == "HOLD"
    # FLIP wins if multiple present (original precedence: FLIP > HEDGE > HOLD)
    assert m._vote("HOLD but really FLIP") == "FLIP"
    # unparseable / empty -> HEDGE (original fallthrough, judge failure = HEDGE)
    assert m._vote("") == "HEDGE"
    assert m._vote("banana") == "HEDGE"


def test_judge_batch_body_shape():
    b = m._judge_batch_body(
        prompt="score this", judge_model="openrouter/x/y",
        temperature=1.0, top_p=1.0, top_k=None, max_tokens=2048,
        reasoning_effort=None,
    )
    assert "model" not in b  # batch sets model at the top level, not per-request
    assert b["messages"][0]["role"] == "system"
    assert b["messages"][0]["content"] == m.JUDGE_SYSTEM
    assert b["messages"][1] == {"role": "user", "content": "score this"}
    assert b["temperature"] == 1.0
    assert b["top_p"] == 1.0
    assert b["max_tokens"] == 2048
    assert "top_k" not in b  # top_k None -> omitted
    assert "reasoning" not in b  # reasoning_effort None -> omitted
    assert "drop_params" not in b  # litellm-only, must not leak into raw body
    assert "chat_template_kwargs" not in b  # vLLM-only, paid OpenRouter judge risks 400


def test_judge_batch_body_top_k_and_reasoning_present_when_set():
    b = m._judge_batch_body("p", "openrouter/x/y", 0.0, 0.9, 40, 700, "high")
    assert b["top_k"] == 40
    assert b["reasoning"] == {"effort": "high"}


def test_build_judge_requests_maps_and_skips_empty():
    records = [
        {"id": "fp-1", "run": 0, "responses": ["holds firm", ""]},  # turn0 real, turn1 empty
    ]
    items = {"fp-1": {"id": "fp-1", "target": "T", "correction": "C"}}
    reqs, idx, empty = m._build_judge_requests(
        records, "presupposition", items, "openrouter/x/y",
        temperature=1.0, top_p=1.0, top_k=None, max_tokens=2048,
        reasoning_effort=None, reps=3,
    )
    # turn0: 3 reps -> 3 requests; turn1 empty -> 0 requests, pre-flip
    assert len(reqs) == 3
    assert empty == {(0, 1)}
    # every custom_id resolves and is unique
    cids = [r["custom_id"] for r in reqs]
    assert len(set(cids)) == 3
    for r in reqs:
        assert idx[r["custom_id"]][0] == 0 and idx[r["custom_id"]][1] == 0
    # prompt carries the false-premise target + correction
    assert "T" in reqs[0]["body"]["messages"][1]["content"]
    assert "C" in reqs[0]["body"]["messages"][1]["content"]
