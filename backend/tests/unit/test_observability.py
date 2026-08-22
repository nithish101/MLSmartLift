import json
import logging

import pytest

from app.observability import (
    JsonFormatter,
    RequestIdFilter,
    _incoming_request_id,
    configure_logging,
    request_id_var,
)


def make_record(msg="hello", **extra):
    record = logging.LogRecord("smartlift.test", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_emits_core_fields_and_extras():
    record = make_record("loaded %s", workout_id=7)
    record.args = ("model",)
    token = request_id_var.set("abc123")
    try:
        RequestIdFilter().filter(record)
    finally:
        request_id_var.reset(token)
    out = json.loads(JsonFormatter().format(record))
    assert out["message"] == "loaded model"
    assert out["level"] == "INFO"
    assert out["logger"] == "smartlift.test"
    assert out["request_id"] == "abc123"
    assert out["workout_id"] == 7
    assert out["timestamp"].endswith("+00:00")


def test_request_id_is_null_outside_a_request():
    record = make_record()
    RequestIdFilter().filter(record)
    assert json.loads(JsonFormatter().format(record))["request_id"] is None


def test_exceptions_are_serialized():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    out = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in out["exception"]


def test_non_json_extras_are_stringified():
    out = json.loads(JsonFormatter().format(make_record(obj=object())))
    assert out["obj"].startswith("<object object")


@pytest.mark.parametrize(
    "header,expected",
    [(b"client-id_1.2", "client-id_1.2"), (b"bad id with spaces", None), (b"x" * 200, None)],
)
def test_incoming_request_id_validation(header, expected):
    assert _incoming_request_id({"headers": [(b"x-request-id", header)]}) == expected
    assert _incoming_request_id({"headers": []}) is None


def test_configure_logging_is_idempotent():
    configure_logging("DEBUG")
    configure_logging("INFO")
    ours = [h for h in logging.getLogger().handlers if h.get_name() == "smartlift-json"]
    assert len(ours) == 1
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("uvicorn.access").disabled
