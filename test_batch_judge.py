import json
import json as _json
import types

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


def test_judge_batch_body_gpt5_family_uses_plain_reasoning_effort():
    # gpt-5 rejects top_k and the {"reasoning": {"effort": ...}} dict form,
    # mirroring _provider_kwargs' gpt-5 branch.
    b = m._judge_batch_body("p", "openai/gpt-5-mini", 0.0, 0.9, 40, 700, "high")
    assert b["reasoning_effort"] == "high"
    assert "top_k" not in b
    assert "reasoning" not in b


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


def test_assign_batch_labels_votes_and_fills_empty():
    records = [{"id": "fp-1", "run": 0, "responses": ["real", ""], "labels": None}]
    # turn0 got 3 judge reps voting HOLD, HOLD, FLIP -> majority HOLD
    def res(cid, word):
        return {"custom_id": cid,
                "response": {"body": {"choices": [{"message": {"content": word}}]}}}
    results = [res("j0", "HOLD"), res("j1", "HOLD"), res("j2", "FLIP")]
    index_map = {"j0": (0, 0, 0), "j1": (0, 0, 1), "j2": (0, 0, 2)}
    empty_flips = {(0, 1)}
    m._assign_batch_labels(records, results, index_map, empty_flips)
    assert records[0]["labels"] == ["HOLD", "FLIP"]  # turn0 majority HOLD, turn1 empty->FLIP


def test_assign_batch_labels_errored_item_is_hedge():
    records = [{"id": "fp-1", "run": 0, "responses": ["real"], "labels": None}]
    results = [{"custom_id": "j0", "error": {"message": "boom"}}]
    m._assign_batch_labels(records, results, {"j0": (0, 0, 0)}, set())
    assert records[0]["labels"] == ["HEDGE"]  # judge failure -> HEDGE, not FLIP


def test_result_text_malformed_but_present_response_is_empty():
    assert m._result_text({"custom_id": "j0", "response": {"body": {"choices": []}}}) == ""
    assert m._result_text({"custom_id": "j0", "response": {"body": {}}}) == ""


def test_assign_batch_labels_malformed_response_is_hedge():
    records = [{"id": "fp-1", "run": 0, "responses": ["real"], "labels": None}]
    results = [{"custom_id": "j0", "response": {"body": {"choices": []}}}]
    m._assign_batch_labels(records, results, {"j0": (0, 0, 0)}, set())
    assert records[0]["labels"] == ["HEDGE"]  # malformed like a failure -> HEDGE, not FLIP


def test_chunk_splits_by_cap():
    reqs = [{"custom_id": f"j{i}"} for i in range(45)]
    chunks = m._chunk(reqs, 20)
    assert [len(c) for c in chunks] == [20, 20, 5]


def test_openrouter_run_judge_batches_reattaches_from_sidecar(tmp_path, monkeypatch):
    sidecar = tmp_path / "presupposition_judge_batch.json"
    sidecar.write_text(json.dumps({"provider": "openrouter", "batch_ids": ["batch_A", "batch_B"]}), encoding="utf-8")
    submitted = []
    monkeypatch.setattr(m, "_batch_submit", lambda *a, **k: submitted.append(a) or "SHOULD_NOT")
    polled = []
    monkeypatch.setattr(m, "_batch_poll",
                        lambda bid, **k: polled.append(bid) or [{"custom_id": bid}])
    out = m._openrouter_run_judge_batches("openrouter/x/y", [{"custom_id": "j0"}], sidecar)
    assert submitted == []                 # reattached, did not resubmit
    assert polled == ["batch_A", "batch_B"]
    assert out == [{"custom_id": "batch_A"}, {"custom_id": "batch_B"}]


def test_openrouter_run_judge_batches_submits_and_writes_sidecar(tmp_path, monkeypatch):
    sidecar = tmp_path / "presupposition_judge_batch.json"
    ids = iter(["batch_1", "batch_2"])
    monkeypatch.setattr(m, "_batch_submit", lambda model, chunk, **k: next(ids))
    monkeypatch.setattr(m, "_batch_poll", lambda bid, **k: [{"custom_id": bid}])
    monkeypatch.setattr(m, "_BATCH_MAX", 1)  # force 2 chunks
    reqs = [{"custom_id": "j0"}, {"custom_id": "j1"}]
    out = m._openrouter_run_judge_batches("openrouter/x/y", reqs, sidecar)
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved == {"provider": "openrouter", "batch_ids": ["batch_1", "batch_2"]}
    assert {r["custom_id"] for r in out} == {"batch_1", "batch_2"}


def _base_args(output_dir, **overrides):
    ns = types.SimpleNamespace(
        model="test/model", judge_model="openrouter/test/judge", api_base=None, judge_api_base=None,
        seed=0, n_items=0, runs=1, max_turns=1, judge_reps=1,
        temperature=0.0, top_p=1.0, top_k=None, max_tokens=64,
        enable_thinking=False, reasoning_effort=None,
        judge_temperature=None, judge_top_p=1.0, judge_top_k=None, judge_max_tokens=64,
        judge_enable_thinking=False, judge_reasoning_effort=None,
        judge_batch=True, resume=False, output_dir=str(output_dir), workers=1,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_run_setting_overwrite_clears_stale_sidecar(tmp_path, monkeypatch):
    """A fresh (non-resume) run must not silently reattach to an old batch
    left behind by a prior crashed/killed --judge-batch run (CLAUDE.md rule 10
    finding: --seed-fixed custom_ids regenerate identically, so a surviving
    sidecar would map stale judge verdicts onto brand-new generations)."""
    item = {"id": "d-1", "system": "sys", "turns": ["u1"], "question": "Q", "target": "T"}
    monkeypatch.setitem(m.LOADERS, "debate", lambda: [item])
    monkeypatch.setattr(m, "run_conversation", lambda *a, **k: (["reply"], [None]))

    args = _base_args(tmp_path)
    outdir = tmp_path / "test_model"
    outdir.mkdir(parents=True)
    sidecar = outdir / "debate_judge_batch.json"
    sidecar.write_text(json.dumps({"provider": "openrouter", "batch_ids": ["STALE_BATCH"]}), encoding="utf-8")
    # A stray transcript, as if a prior run crashed mid-batch-poll.
    (outdir / "debate_transcripts.jsonl").write_text('{"id":"d-1","run":0}\n', encoding="utf-8")

    submitted_chunks = {}

    def fake_submit(model, chunk, **k):
        bid = f"NEW_{len(submitted_chunks)}"
        submitted_chunks[bid] = chunk
        return bid

    def fake_poll(bid, **k):
        # If a reattach to the stale sidecar happened, this would be called
        # with "STALE_BATCH", which was never submitted here -> KeyError.
        chunk = submitted_chunks[bid]
        return [{"custom_id": r["custom_id"],
                  "response": {"body": {"choices": [{"message": {"content": "HOLD"}}]}}}
                for r in chunk]

    monkeypatch.setattr(m, "_batch_submit", fake_submit)
    monkeypatch.setattr(m, "_batch_poll", fake_poll)

    m.run_setting("debate", args)

    assert list(submitted_chunks.keys()) == ["NEW_0"]  # fresh submit, no reattach
    assert not sidecar.exists()  # cleared up front; removed again on completion


def test_run_setting_resume_preserves_sidecar(tmp_path, monkeypatch):
    """--resume must keep reattaching to a batch already in flight for this
    run identity — that is the intended double-charge-avoidance behaviour."""
    monkeypatch.setattr(m, "run_conversation",
                         lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not regenerate")))
    monkeypatch.setattr(m, "run_judge_batches",
                         lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not batch-judge; nothing pending")))

    args = _base_args(tmp_path, resume=True)
    outdir = tmp_path / "test_model"
    outdir.mkdir(parents=True)
    sidecar = outdir / "debate_judge_batch.json"
    sidecar.write_text(json.dumps({"provider": "openrouter", "batch_ids": ["OLD_BATCH"]}), encoding="utf-8")

    record = {"id": "d-1", "run": 0, "question": "Q", "target": "T", "meta": {},
              "responses": ["reply"], "diagnostics": [None], "labels": ["HOLD"]}
    (outdir / "debate_transcripts.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    meta = {
        "model": args.model, "judge_model": args.judge_model, "setting": "debate",
        "seed": args.seed, "n_items": args.n_items, "runs": args.runs,
        "max_turns": args.max_turns, "judge_reps": args.judge_reps, "item_ids": ["d-1"],
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "max_tokens": args.max_tokens, "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
        "judge_temperature": args.judge_temperature, "judge_top_p": args.judge_top_p,
        "judge_top_k": args.judge_top_k, "judge_max_tokens": args.judge_max_tokens,
        "judge_enable_thinking": args.judge_enable_thinking,
        "judge_reasoning_effort": args.judge_reasoning_effort,
        "judge_batch": args.judge_batch,
    }
    (outdir / "debate_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Item lookup only; the single (id, run) pair is already "done" on disk.
    monkeypatch.setitem(m.LOADERS, "debate",
                         lambda: [{"id": "d-1", "system": "sys", "turns": ["u1"],
                                   "question": "Q", "target": "T"}])

    m.run_setting("debate", args)

    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"provider": "openrouter", "batch_ids": ["OLD_BATCH"]}


def test_openai_input_lines_injects_model_and_wraps():
    reqs = [{"custom_id": "j0", "body": {"messages": [{"role": "user", "content": "hi"}],
                                          "temperature": 0.0, "max_tokens": 5}}]
    raw = m._openai_input_lines(reqs, "openai/gpt-5.6-luna")
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 1
    obj = _json.loads(lines[0])
    assert obj["custom_id"] == "j0"
    assert obj["method"] == "POST"
    assert obj["url"] == "/v1/chat/completions"
    # openai/ prefix stripped for the API's model field
    assert obj["body"]["model"] == "gpt-5.6-luna"
    # original body fields preserved verbatim (methodology parity)
    assert obj["body"]["messages"] == reqs[0]["body"]["messages"]
    assert obj["body"]["temperature"] == 0.0
    assert obj["body"]["max_tokens"] == 5

def test_openai_input_lines_unprefixed_model_kept():
    raw = m._openai_input_lines([{"custom_id": "j0", "body": {"messages": []}}], "gpt-5.6-luna")
    assert _json.loads(raw.decode())["body"]["model"] == "gpt-5.6-luna"


class _FakeResp:
    def __init__(self, payload_bytes): self._b = payload_bytes
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False

def _patch_urlopen(monkeypatch, handler):
    """handler(req) -> bytes; req is the urllib.request.Request."""
    monkeypatch.setattr(m.urllib.request, "urlopen", lambda req, *a, **k: _FakeResp(handler(req)))

def test_openai_headers_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import pytest
    with pytest.raises(SystemExit):
        m._openai_headers()

def test_openai_upload_file_multipart(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}
    def handler(req):
        seen["url"] = req.full_url
        seen["ctype"] = req.headers.get("Content-type") or req.headers.get("Content-Type")
        seen["body"] = req.data
        return b'{"id": "file-abc"}'
    _patch_urlopen(monkeypatch, handler)
    fid = m._openai_upload_file(b'{"a":1}\n{"b":2}')
    assert fid == "file-abc"
    assert seen["url"].endswith("/v1/files")
    assert "multipart/form-data; boundary=" in seen["ctype"]
    body = seen["body"]
    assert b'name="purpose"' in body and b'batch' in body
    assert b'name="file"; filename="judge_batch.jsonl"' in body
    assert b'{"a":1}\n{"b":2}' in body   # payload embedded intact

def test_openai_batch_create(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}
    def handler(req):
        seen["url"] = req.full_url; seen["body"] = _json.loads(req.data)
        return b'{"id": "batch_1", "status": "validating"}'
    _patch_urlopen(monkeypatch, handler)
    bid = m._openai_batch_create("file-abc")
    assert bid == "batch_1"
    assert seen["url"].endswith("/v1/batches")
    assert seen["body"] == {"input_file_id": "file-abc",
                            "endpoint": "/v1/chat/completions",
                            "completion_window": "24h"}

def test_openai_batch_poll_completed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _patch_urlopen(monkeypatch, lambda req: b'{"status":"completed","output_file_id":"file-out","error_file_id":"file-err"}')
    out, err = m._openai_batch_poll("batch_1", interval=0)
    assert out == "file-out" and err == "file-err"

def test_openai_batch_poll_failed_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _patch_urlopen(monkeypatch, lambda req: b'{"status":"failed"}')
    import pytest
    with pytest.raises(RuntimeError):
        m._openai_batch_poll("batch_1", interval=0)

def test_openai_download_parses_jsonl_into_result_items(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    line = _json.dumps({"custom_id": "j0",
                        "response": {"body": {"choices": [{"message": {"content": "FLIP"}}]}}})
    _patch_urlopen(monkeypatch, lambda req: (line + "\n").encode())
    items = m._openai_download("file-out")
    assert len(items) == 1
    # shape is exactly what _result_text already consumes
    assert m._result_text(items[0]) == "FLIP"

def test_openai_download_none_returns_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert m._openai_download(None) == []


def _one_req(cid="j0"):
    return {"custom_id": cid, "body": {"messages": [{"role": "user", "content": "x"}],
                                       "temperature": 0.0, "max_tokens": 5}}


def test_openai_run_submits_and_merges_output_and_error(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = []
    monkeypatch.setattr(m, "_openai_upload_file", lambda b, **k: (calls.append("upload"), "file-in")[1])
    monkeypatch.setattr(m, "_openai_batch_create", lambda fid: (calls.append(("create", fid)), "batch_1")[1])
    monkeypatch.setattr(m, "_openai_batch_poll", lambda bid, **k: ("file-out", "file-err"))
    def fake_dl(fid):
        if fid == "file-out":
            return [{"custom_id": "j0", "response": {"body": {"choices": [{"message": {"content": "HOLD"}}]}}}]
        if fid == "file-err":
            return [{"custom_id": "j1", "error": {"message": "boom"}}]  # failed judge -> HEDGE
        return []
    monkeypatch.setattr(m, "_openai_download", fake_dl)
    sidecar = tmp_path / "debate_judge_batch.json"
    results = m._openai_run_judge_batches("openai/gpt-5.6-luna", [_one_req("j0"), _one_req("j1")], sidecar)
    assert {r["custom_id"] for r in results} == {"j0", "j1"}      # output + error merged
    assert _json.loads(sidecar.read_text(encoding="utf-8")) == {"provider": "openai", "batch_ids": ["batch_1"]}


def test_openai_run_reattaches_from_sidecar_without_resubmitting(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sidecar = tmp_path / "debate_judge_batch.json"
    sidecar.write_text(_json.dumps({"provider": "openai", "batch_ids": ["batch_existing"]}), encoding="utf-8")
    def boom(*a, **k): raise AssertionError("must not re-upload/re-create on reattach")
    monkeypatch.setattr(m, "_openai_upload_file", boom)
    monkeypatch.setattr(m, "_openai_batch_create", boom)
    polled = []
    monkeypatch.setattr(m, "_openai_batch_poll", lambda bid, **k: (polled.append(bid), ("file-out", None))[1])
    monkeypatch.setattr(m, "_openai_download", lambda fid: [] if fid is None else
                        [{"custom_id": "j0", "response": {"body": {"choices": [{"message": {"content": "FLIP"}}]}}}])
    results = m._openai_run_judge_batches("openai/gpt-5.6-luna", [_one_req()], sidecar)
    assert polled == ["batch_existing"]
    assert results[0]["custom_id"] == "j0"


def test_openai_run_rejects_wrong_provider_sidecar(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sidecar = tmp_path / "debate_judge_batch.json"
    sidecar.write_text(_json.dumps({"provider": "openrouter", "batch_ids": ["x"]}), encoding="utf-8")
    import pytest
    with pytest.raises(RuntimeError):
        m._openai_run_judge_batches("openai/gpt-5.6-luna", [_one_req()], sidecar)


def test_openrouter_run_rejects_wrong_provider_sidecar(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    sidecar = tmp_path / "debate_judge_batch.json"
    sidecar.write_text(_json.dumps({"provider": "openai", "batch_ids": ["x"]}), encoding="utf-8")
    import pytest
    with pytest.raises(RuntimeError):
        m._openrouter_run_judge_batches("openrouter/x/y", [_one_req()], sidecar)


def test_batch_provider_by_prefix():
    assert m._batch_provider("openrouter/openai/gpt-5") == "openrouter"
    assert m._batch_provider("openai/gpt-5.6-luna") == "openai"
    assert m._batch_provider("gpt-5.6-luna") == "openai"          # unprefixed -> openai (litellm default)


def test_run_judge_batches_dispatches(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(m, "_openrouter_run_judge_batches",
                        lambda model, reqs, sc: (seen.__setitem__("who", "or"), [])[1])
    monkeypatch.setattr(m, "_openai_run_judge_batches",
                        lambda model, reqs, sc: (seen.__setitem__("who", "oa"), [])[1])
    m.run_judge_batches("openrouter/x/y", [], tmp_path / "s.json")
    assert seen["who"] == "or"
    m.run_judge_batches("openai/gpt-5.6-luna", [], tmp_path / "s.json")
    assert seen["who"] == "oa"


# --------------------------------------------------------------------------
# Adversarial & Edge-Case Tests
# --------------------------------------------------------------------------

def test_openai_input_lines_unicode_handling():
    reqs = [
        {"custom_id": "j0", "body": {"messages": [{"role": "user", "content": "你好 世界 🚀 — em-dash"}],
                                      "temperature": 0.0, "max_tokens": 5}},
    ]
    raw = m._openai_input_lines(reqs, "openai/gpt-5.6-luna")
    decoded = raw.decode("utf-8")
    assert "你好 世界 🚀 — em-dash" in decoded
    obj = _json.loads(decoded.splitlines()[0])
    assert obj["body"]["messages"][0]["content"] == "你好 世界 🚀 — em-dash"


def test_openai_upload_file_with_unicode_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}
    def handler(req):
        seen["url"] = req.full_url
        seen["ctype"] = req.headers.get("Content-type") or req.headers.get("Content-Type")
        seen["body"] = req.data
        return b'{"id": "file-unicode-123"}'
    _patch_urlopen(monkeypatch, handler)

    payload = "你好 世界 🚀".encode("utf-8")
    fid = m._openai_upload_file(payload)
    assert fid == "file-unicode-123"
    assert payload in seen["body"]
    assert seen["ctype"].startswith("multipart/form-data; boundary=")


def test_openai_batch_poll_transitions_and_completes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    responses = iter([
        b'{"status": "validating", "request_counts": {"completed": 0, "total": 10}}',
        b'{"status": "in_progress", "request_counts": {"completed": 5, "total": 10}}',
        b'{"status": "finalizing", "request_counts": {"completed": 10, "total": 10}}',
        b'{"status": "completed", "output_file_id": "file-out-final", "error_file_id": null}',
    ])
    monkeypatch.setattr(m.time, "sleep", lambda sec: None)
    _patch_urlopen(monkeypatch, lambda req: next(responses))
    out, err = m._openai_batch_poll("batch_multi_step", interval=0)
    assert out == "file-out-final"
    assert err is None


def test_openai_batch_poll_expired_and_cancelled_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for bad_status in ("expired", "cancelled"):
        _patch_urlopen(monkeypatch, lambda req, st=bad_status: f'{{"status":"{st}"}}'.encode())
        import pytest
        with pytest.raises(RuntimeError, match=f"openai batch_test ended {bad_status}"):
            m._openai_batch_poll("batch_test", interval=0)


def test_openai_download_with_blank_lines_and_unicode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    line1 = _json.dumps({"custom_id": "j0",
                         "response": {"body": {"choices": [{"message": {"content": "HOLD — 保持"}}]}}})
    line2 = _json.dumps({"custom_id": "j1",
                         "response": {"body": {"choices": [{"message": {"content": "FLIP"}}]}}})
    raw_content = f"\n\n{line1}\n  \n{line2}\n\n".encode("utf-8")
    _patch_urlopen(monkeypatch, lambda req: raw_content)
    items = m._openai_download("file-out-unicode")
    assert len(items) == 2
    assert m._result_text(items[0]) == "HOLD — 保持"
    assert m._result_text(items[1]) == "FLIP"


def test_openai_run_judge_batches_multi_chunk(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(m, "_OPENAI_BATCH_MAX", 2)  # Cap chunk at 2

    uploaded_files = []
    created_batches = []
    polled_batches = []

    monkeypatch.setattr(m, "_openai_upload_file",
                        lambda b, **k: (uploaded_files.append(b), f"file-{len(uploaded_files)}")[1])
    monkeypatch.setattr(m, "_openai_batch_create",
                        lambda fid: (created_batches.append(fid), f"batch-{len(created_batches)}")[1])
    monkeypatch.setattr(m, "_openai_batch_poll",
                        lambda bid, **k: (polled_batches.append(bid), (f"out-{bid}", f"err-{bid}"))[1])

    def fake_dl(fid):
        if fid.startswith("out-"):
            bid = fid.split("out-")[1]
            return [{"custom_id": f"res-{bid}"}]
        return []

    monkeypatch.setattr(m, "_openai_download", fake_dl)

    # 5 requests -> 3 chunks (2, 2, 1)
    reqs = [_one_req(f"r{i}") for i in range(5)]
    sidecar = tmp_path / "debate_judge_batch.json"

    results = m._openai_run_judge_batches("openai/gpt-5.6-luna", reqs, sidecar)

    assert len(uploaded_files) == 3
    assert created_batches == ["file-1", "file-2", "file-3"]
    assert polled_batches == ["batch-1", "batch-2", "batch-3"]
    assert [r["custom_id"] for r in results] == ["res-batch-1", "res-batch-2", "res-batch-3"]
    saved_meta = _json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved_meta == {"provider": "openai", "batch_ids": ["batch-1", "batch-2", "batch-3"]}


def test_sidecar_missing_provider_tag_rejected(monkeypatch, tmp_path):
    # Old legacy sidecar without "provider" key must be rejected
    legacy_sidecar = tmp_path / "legacy_judge_batch.json"
    legacy_sidecar.write_text(_json.dumps({"batch_ids": ["old_batch"]}), encoding="utf-8")

    import pytest
    with pytest.raises(RuntimeError, match="provider None, not 'openai'"):
        m._openai_run_judge_batches("openai/gpt-5.6-luna", [_one_req()], legacy_sidecar)

    with pytest.raises(RuntimeError, match="provider None, not 'openrouter'"):
        m._openrouter_run_judge_batches("openrouter/x/y", [_one_req()], legacy_sidecar)


def test_assign_batch_labels_unknown_custom_id_ignored():
    records = [{"id": "fp-1", "run": 0, "responses": ["real"], "labels": None}]
    results = [
        {"custom_id": "unknown_id", "response": {"body": {"choices": [{"message": {"content": "FLIP"}}]}}},
        {"custom_id": "j0", "response": {"body": {"choices": [{"message": {"content": "HOLD"}}]}}},
    ]
    index_map = {"j0": (0, 0, 0)}
    m._assign_batch_labels(records, results, index_map, set())
    assert records[0]["labels"] == ["HOLD"]


def test_run_setting_openai_judge_batch_end_to_end(tmp_path, monkeypatch):
    """End-to-end run_setting with openai judge-batch."""
    item = {"id": "d-1", "system": "sys", "turns": ["u1"], "question": "Q", "target": "T"}
    monkeypatch.setitem(m.LOADERS, "debate", lambda: [item])
    monkeypatch.setattr(m, "run_conversation", lambda *a, **k: (["reply"], [None]))

    args = _base_args(tmp_path, judge_model="openai/gpt-5.6-luna")
    outdir = tmp_path / "test_model"

    uploaded = []
    created = []
    polled = []

    monkeypatch.setattr(m, "_openai_upload_file", lambda b, **k: (uploaded.append(b), "file-in-1")[1])
    monkeypatch.setattr(m, "_openai_batch_create", lambda fid: (created.append(fid), "batch-1")[1])
    monkeypatch.setattr(m, "_openai_batch_poll", lambda bid, **k: (polled.append(bid), ("file-out-1", None))[1])
    monkeypatch.setattr(m, "_openai_download", lambda fid: [{"custom_id": "j0",
                                                             "response": {"body": {"choices": [{"message": {"content": "HOLD"}}]}}}] if fid == "file-out-1" else [])

    summary = m.run_setting("debate", args)

    assert len(uploaded) == 1
    assert created == ["file-in-1"]
    assert polled == ["batch-1"]
    assert summary["label_distribution"] == {"HOLD": 1}

    # Sidecar should be removed after completion
    sidecar = outdir / "debate_judge_batch.json"
    assert not sidecar.exists()

    # Transcripts should have labels filled
    transcripts = (outdir / "debate_transcripts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(transcripts) == 1
    rec = _json.loads(transcripts[0])
    assert rec["labels"] == ["HOLD"]


def test_run_setting_openai_judge_batch_resume_reattach(tmp_path, monkeypatch):
    """End-to-end run_setting --resume reattaching to in-flight OpenAI batch."""
    monkeypatch.setattr(m, "run_conversation",
                         lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not regenerate")))
    def boom(*a, **k): raise AssertionError("should not upload/create new batch on reattach")
    monkeypatch.setattr(m, "_openai_upload_file", boom)
    monkeypatch.setattr(m, "_openai_batch_create", boom)

    polled = []
    monkeypatch.setattr(m, "_openai_batch_poll", lambda bid, **k: (polled.append(bid), ("file-out-res", None))[1])
    monkeypatch.setattr(m, "_openai_download", lambda fid: [{"custom_id": "j0",
                                                             "response": {"body": {"choices": [{"message": {"content": "FLIP"}}]}}}] if fid == "file-out-res" else [])

    args = _base_args(tmp_path, judge_model="openai/gpt-5.6-luna", resume=True)
    outdir = tmp_path / "test_model"
    outdir.mkdir(parents=True)

    sidecar = outdir / "debate_judge_batch.json"
    sidecar.write_text(_json.dumps({"provider": "openai", "batch_ids": ["batch_inflight"]}), encoding="utf-8")

    # Transcript was written with labels: None before the crash/kill
    record = {"id": "d-1", "run": 0, "question": "Q", "target": "T", "meta": {},
              "responses": ["reply"], "diagnostics": [None], "labels": None}
    (outdir / "debate_transcripts.jsonl").write_text(_json.dumps(record) + "\n", encoding="utf-8")
    meta = {
        "model": args.model, "judge_model": args.judge_model, "setting": "debate",
        "seed": args.seed, "n_items": args.n_items, "runs": args.runs,
        "max_turns": args.max_turns, "judge_reps": args.judge_reps, "item_ids": ["d-1"],
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "max_tokens": args.max_tokens, "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
        "judge_temperature": args.judge_temperature, "judge_top_p": args.judge_top_p,
        "judge_top_k": args.judge_top_k, "judge_max_tokens": args.judge_max_tokens,
        "judge_enable_thinking": args.judge_enable_thinking,
        "judge_reasoning_effort": args.judge_reasoning_effort,
        "judge_batch": args.judge_batch,
    }
    (outdir / "debate_run_meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")

    monkeypatch.setitem(m.LOADERS, "debate",
                         lambda: [{"id": "d-1", "system": "sys", "turns": ["u1"],
                                   "question": "Q", "target": "T"}])

    summary = m.run_setting("debate", args)

    assert polled == ["batch_inflight"]
    assert summary["label_distribution"] == {"FLIP": 1}
    assert not sidecar.exists()  # deleted after successful completion


def test_judge_batch_body_unprefixed_gpt5_and_openai_reasoning_models():
    # Unprefixed gpt-5 model should use plain reasoning_effort and no top_k or reasoning dict
    b1 = m._judge_batch_body("p", "gpt-5.6-luna", 0.0, 0.9, 40, 700, "high")
    assert b1["reasoning_effort"] == "high"
    assert "top_k" not in b1
    assert "reasoning" not in b1

    # openai/o3-mini should route as openai provider and use plain reasoning_effort
    b2 = m._judge_batch_body("p", "openai/o3-mini", 0.0, 0.9, 40, 700, "medium")
    assert b2["reasoning_effort"] == "medium"
    assert "top_k" not in b2
    assert "reasoning" not in b2


def test_provider_kwargs_unprefixed_and_openai_models():
    # Unprefixed gpt-5 model should omit extra_body and reasoning dict
    kw1 = m._provider_kwargs("gpt-5.6-luna", top_k=40, enable_thinking=False, reasoning_effort="high")
    assert kw1["reasoning_effort"] == "high"
    assert "extra_body" not in kw1
    assert "reasoning" not in kw1

    # openai/gpt-5.6-luna should omit extra_body and reasoning dict
    kw2 = m._provider_kwargs("openai/gpt-5.6-luna", top_k=40, enable_thinking=False, reasoning_effort="high")
    assert kw2["reasoning_effort"] == "high"
    assert "extra_body" not in kw2
    assert "reasoning" not in kw2

    # openrouter model should retain extra_body and reasoning dict
    kw3 = m._provider_kwargs("openrouter/poolside/laguna-xs-2.1:free", top_k=40, enable_thinking=False, reasoning_effort="high")
    assert kw3["reasoning_effort"] == "high"
    assert kw3["extra_body"]["top_k"] == 40
    assert kw3["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert kw3["reasoning"] == {"effort": "high"}


def test_openai_batch_poll_failed_includes_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    err_obj = {"object": "list", "data": [{"code": "invalid_request", "message": "Model not eligible"}]}
    _patch_urlopen(monkeypatch, lambda req: f'{{"status":"failed","errors":{_json.dumps(err_obj)}}}'.encode())
    import pytest
    with pytest.raises(RuntimeError, match="Model not eligible"):
        m._openai_batch_poll("batch_fail_with_errs", interval=0)


def test_run_setting_openai_judge_batch_all_empty_responses(tmp_path, monkeypatch):
    """When all responses are empty, judge batching should not upload any batch and all score FLIP."""
    item = {"id": "d-1", "system": "sys", "turns": ["u1"], "question": "Q", "target": "T"}
    monkeypatch.setitem(m.LOADERS, "debate", lambda: [item])
    # Empty responses from model
    monkeypatch.setattr(m, "run_conversation", lambda *a, **k: (["", ""], [None, None]))

    def boom(*a, **k): raise AssertionError("should not upload/poll when all responses are empty")
    monkeypatch.setattr(m, "_openai_upload_file", boom)
    monkeypatch.setattr(m, "_openai_batch_create", boom)
    monkeypatch.setattr(m, "_openai_batch_poll", boom)

    args = _base_args(tmp_path, judge_model="openai/gpt-5.6-luna", max_turns=2)
    outdir = tmp_path / "test_model"

    summary = m.run_setting("debate", args)
    assert summary["label_distribution"] == {"FLIP": 2}
    transcripts = (outdir / "debate_transcripts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = _json.loads(transcripts[0])
    assert rec["labels"] == ["FLIP", "FLIP"]


def test_run_setting_openai_judge_batch_all_errored_requests_hedge(tmp_path, monkeypatch):
    """When OpenAI batch returns only error_file_id, all errored judge requests must score HEDGE."""
    item = {"id": "d-1", "system": "sys", "turns": ["u1"], "question": "Q", "target": "T"}
    monkeypatch.setitem(m.LOADERS, "debate", lambda: [item])
    monkeypatch.setattr(m, "run_conversation", lambda *a, **k: (["reply1", "reply2"], [None, None]))

    monkeypatch.setattr(m, "_openai_upload_file", lambda b, **k: "file-in-err")
    monkeypatch.setattr(m, "_openai_batch_create", lambda fid: "batch-err-only")
    monkeypatch.setattr(m, "_openai_batch_poll", lambda bid, **k: (None, "file-err-only"))
    monkeypatch.setattr(m, "_openai_download", lambda fid: [
        {"custom_id": "j0", "error": {"message": "rate limited"}},
        {"custom_id": "j1", "error": {"message": "rate limited"}},
    ] if fid == "file-err-only" else [])

    args = _base_args(tmp_path, judge_model="openai/gpt-5.6-luna", max_turns=2)
    outdir = tmp_path / "test_model"

    summary = m.run_setting("debate", args)
    assert summary["label_distribution"] == {"HEDGE": 2}
    transcripts = (outdir / "debate_transcripts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = _json.loads(transcripts[0])
    assert rec["labels"] == ["HEDGE", "HEDGE"]


def test_run_setting_openai_judge_batch_reps_majority_vote(tmp_path, monkeypatch):
    """Test judge-reps=3 majority voting with OpenAI batch in run_setting."""
    item = {"id": "d-1", "system": "sys", "turns": ["u1"], "question": "Q", "target": "T"}
    monkeypatch.setitem(m.LOADERS, "debate", lambda: [item])
    monkeypatch.setattr(m, "run_conversation", lambda *a, **k: (["reply1"], [None]))

    created_lines = []
    def fake_upload(b, **k):
        created_lines.extend(b.decode("utf-8").strip().splitlines())
        return "file-in-reps"

    monkeypatch.setattr(m, "_openai_upload_file", fake_upload)
    monkeypatch.setattr(m, "_openai_batch_create", lambda fid: "batch-reps")
    monkeypatch.setattr(m, "_openai_batch_poll", lambda bid, **k: ("file-out-reps", None))

    # 3 reps for turn 0: j0->HOLD, j1->HOLD, j2->FLIP => Majority HOLD
    monkeypatch.setattr(m, "_openai_download", lambda fid: [
        {"custom_id": "j0", "response": {"body": {"choices": [{"message": {"content": "HOLD"}}]}}},
        {"custom_id": "j1", "response": {"body": {"choices": [{"message": {"content": "HOLD"}}]}}},
        {"custom_id": "j2", "response": {"body": {"choices": [{"message": {"content": "FLIP"}}]}}},
    ] if fid == "file-out-reps" else [])

    args = _base_args(tmp_path, judge_model="openai/gpt-5.6-luna", judge_reps=3, max_turns=1)
    summary = m.run_setting("debate", args)

    assert len(created_lines) == 3
    assert summary["label_distribution"] == {"HOLD": 1}


def test_batch_provider_names_other_providers():
    # Not a binary openrouter/openai split: a third provider must name itself so
    # run_judge_batches can reject it instead of POSTing it to api.openai.com.
    assert m._batch_provider("anthropic/claude-sonnet-4-5") == "anthropic"
    assert m._batch_provider("gemini/gemini-3-pro") == "gemini"


def test_run_judge_batches_rejects_unsupported_provider(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="only supports openrouter/ and openai/"):
        m.run_judge_batches("anthropic/claude-sonnet-4-5", [], tmp_path / "s.json")


def test_provider_kwargs_direct_non_openai_provider_keeps_extra_body():
    # anthropic/ is the judge SETUP.md recommends; it must keep the pre-existing
    # extra_body/reasoning-dict path, not get swept into the OpenAI branch.
    kw = m._provider_kwargs("anthropic/claude-sonnet-4-5", top_k=40,
                            enable_thinking=False, reasoning_effort="high")
    assert kw["extra_body"]["top_k"] == 40
    assert kw["reasoning"] == {"effort": "high"}


def test_judge_batch_body_openrouter_gpt5_named_clone_keeps_top_k():
    # "gpt-5" in an OpenRouter slug does not make it OpenAI; an open-weight model
    # served by OpenRouter still takes top_k and the reasoning dict.
    b = m._judge_batch_body("p", "openrouter/z-ai/gpt-5-clone", 0.0, 0.9, 40, 700, "high")
    assert b["top_k"] == 40
    assert b["reasoning"] == {"effort": "high"}


def test_provider_kwargs_and_judge_batch_body_agree_on_provider():
    # Sync and batch judges must branch identically, or the same --judge-model
    # scores differently with and without --judge-batch.
    for model in ["openai/gpt-5.6-luna", "gpt-5.6-luna", "openrouter/openai/gpt-5",
                  "openrouter/z-ai/gpt-5-clone", "openrouter/x/y",
                  "anthropic/claude-sonnet-4-5"]:
        kw = m._provider_kwargs(model, top_k=40, enable_thinking=False, reasoning_effort="high")
        b = m._judge_batch_body("p", model, 0.0, 0.9, 40, 700, "high")
        assert ("extra_body" in kw) == ("top_k" in b), model
        assert ("reasoning" in kw) == ("reasoning" in b), model
