"""Tests for the shared HTTP probes."""

from __future__ import annotations

import httpx
import pytest

from llm.core import http


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://x")
            raise httpx.HTTPStatusError(
                "boom", request=request, response=httpx.Response(self.status_code, request=request)
            )

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class TestIsHealthy:
    def test_true_on_200(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp(200))
        assert http.is_healthy("http://x/health") is True

    def test_false_on_non_200(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp(503))
        assert http.is_healthy("http://x/health") is False

    def test_false_when_unreachable(self, monkeypatch):
        def _boom(*a, **kw):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", _boom)
        assert http.is_healthy("http://x/health") is False

    def test_false_on_timeout(self, monkeypatch):
        def _boom(*a, **kw):
            raise httpx.ReadTimeout("slow")

        monkeypatch.setattr(httpx, "get", _boom)
        assert http.is_healthy("http://x/health") is False


class TestWaitUntilHealthy:
    def test_returns_immediately_when_already_healthy(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp(200))
        slept = []
        assert http.wait_until_healthy("http://x", timeout=5, sleep=slept.append) is True
        assert slept == []

    def test_polls_until_the_server_comes_up(self, monkeypatch):
        codes = iter([503, 503, 200])
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp(next(codes)))
        slept = []
        assert http.wait_until_healthy("http://x", timeout=5, sleep=slept.append) is True
        assert len(slept) == 2

    def test_gives_up_after_the_timeout(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp(503))
        assert http.wait_until_healthy("http://x", timeout=0, sleep=lambda _: None) is False


class TestGetJson:
    def test_returns_the_decoded_body(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp(200, payload={"n_ctx": 4096}))
        assert http.get_json("http://x/model_info") == {"n_ctx": 4096}

    def test_none_on_malformed_json(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp(200, text="<html>"))
        assert http.get_json("http://x/model_info") is None

    def test_none_on_error_status(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp(404))
        assert http.get_json("http://x/model_info") is None

    def test_none_when_unreachable(self, monkeypatch):
        def _boom(*a, **kw):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", _boom)
        assert http.get_json("http://x/model_info") is None


@pytest.mark.parametrize("probe", [http.is_healthy, http.get_json])
def test_probes_never_raise_on_transport_errors(probe, monkeypatch):
    """Callers treat these as best-effort; an exception would abort a command."""

    def _boom(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    probe("http://x")
