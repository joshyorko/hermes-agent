"""Exercise title generation across the real background-thread boundary."""

import contextvars
import threading
from pathlib import Path
from types import SimpleNamespace

from agent import title_generator
from hermes_cli.auth import _auth_file_path
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_state import SessionDB


def test_title_worker_keeps_originating_profile_context(tmp_path, monkeypatch):
    """A multiplexed title must use its caller's config/auth home, not process defaults."""
    default = tmp_path / "process-default"
    profile = default / "profiles" / "research"
    for home, language in ((default, "German"), (profile, "French")):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "auxiliary:\n  title_generation:\n    enabled: true\n"
            f"    language: {language}\n"
        )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default))
    marker = contextvars.ContextVar("title_worker_marker", default="unset")
    seen = []
    workers = []
    finished = threading.Event()
    runtime = {"provider": "openai", "model": "test-model"}

    def model_boundary(**kwargs):
        workers.append(threading.current_thread())
        seen.append((get_hermes_home(), _auth_file_path(),
                     title_generator._title_language(), marker.get(),
                     kwargs["main_runtime"]))
        marker.set("worker-only")
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"title":"Thread context regression"}')
        )])

    def on_title(title, source):
        if source == "llm":
            finished.set()

    monkeypatch.setattr(title_generator, "call_llm", model_boundary)
    db = SessionDB(profile / "state.db")
    db.create_session("title-context", source="cli")
    home_token = set_hermes_home_override(profile)
    marker_token = marker.set("originating-turn")
    try:
        title_generator.maybe_auto_title(
            db, "title-context", "Investigate background thread context",
            main_runtime=runtime, title_callback=on_title,
        )
        assert finished.wait(10), "title worker did not finish"
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive()
        assert seen == [(profile, profile / "auth.json", "French",
                         "originating-turn", runtime)]
        assert marker.get() == "originating-turn"
        assert db.get_session_title_source("title-context") == "llm"
    finally:
        reset_hermes_home_override(home_token)
        marker.reset(marker_token)
        db.close()
