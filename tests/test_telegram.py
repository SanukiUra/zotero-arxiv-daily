"""Tests for the Telegram notifier path: renderer, validator, dispatch."""

import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.construct_email import (
    TELEGRAM_MAX_LEN,
    render_telegram_messages,
)
from zotero_arxiv_daily.executor import Executor, validate_notifier_config
from tests.canned_responses import make_sample_paper


# ---------------------------------------------------------------------------
# render_telegram_messages
# ---------------------------------------------------------------------------


def test_render_telegram_messages_empty():
    msgs = render_telegram_messages([])
    assert len(msgs) == 1
    assert "No Papers Today" in msgs[0]


def test_render_telegram_messages_basic_fields():
    paper = make_sample_paper(score=7.5, tldr="Great paper.", affiliations=["MIT"])
    msgs = render_telegram_messages([paper])
    assert len(msgs) == 1
    body = msgs[0]
    assert "Sample Paper Title" in body
    assert "Great paper." in body
    assert "MIT" in body
    assert "PDF" in body


def test_render_telegram_messages_html_escaped():
    paper = make_sample_paper(title="A <script>alert(1)</script> paper", tldr="<b>bold</b>", score=7.0)
    msgs = render_telegram_messages([paper])
    body = msgs[0]
    assert "<script>" not in body
    assert "&lt;script&gt;" in body
    assert "&lt;b&gt;bold&lt;/b&gt;" in body


def test_render_telegram_messages_chunks_when_over_limit():
    papers = [make_sample_paper(title=f"Paper {i}", tldr="x" * 1000, score=7.0) for i in range(10)]
    msgs = render_telegram_messages(papers)
    assert len(msgs) >= 2
    assert all(len(m) <= TELEGRAM_MAX_LEN for m in msgs)


# ---------------------------------------------------------------------------
# validate_notifier_config
# ---------------------------------------------------------------------------


def test_validate_notifier_telegram_ok(config):
    with open_dict(config):
        config.executor.notifier = "telegram"
    assert validate_notifier_config(config) == "telegram"


def test_validate_notifier_telegram_missing_token(config):
    with open_dict(config):
        config.executor.notifier = "telegram"
        config.telegram.bot_token = None
    with pytest.raises(ValueError, match="bot_token"):
        validate_notifier_config(config)


def test_validate_notifier_email_ok(config):
    with open_dict(config):
        config.executor.notifier = "email"
    assert validate_notifier_config(config) == "email"


def test_validate_notifier_email_missing_sender(config):
    with open_dict(config):
        config.executor.notifier = "email"
        config.email.sender = None
    with pytest.raises(ValueError, match="sender"):
        validate_notifier_config(config)


def test_validate_notifier_unknown_value(config):
    with open_dict(config):
        config.executor.notifier = "carrier-pigeon"
    with pytest.raises(ValueError, match="carrier-pigeon"):
        validate_notifier_config(config)


# ---------------------------------------------------------------------------
# E2E: Executor.run() routes to telegram
# ---------------------------------------------------------------------------


def test_run_end_to_end_telegram(config, monkeypatch):
    from tests.canned_responses import (
        make_sample_paper,
        make_stub_openai_client,
        make_stub_zotero_client,
    )

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        config.executor.notifier = "telegram"

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: [make_sample_paper(title="TG Paper")],
    )

    sent = []

    def fake_send_telegram(cfg, messages):
        sent.append((cfg.telegram.chat_id, list(messages)))

    monkeypatch.setattr("zotero_arxiv_daily.executor.send_telegram", fake_send_telegram)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    Executor(config).run()

    assert len(sent) == 1
    chat_id, messages = sent[0]
    assert str(chat_id) == "12345"
    assert any("TG Paper" in m for m in messages)
