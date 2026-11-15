"""Who the rate limit counts as one caller."""

from starlette.requests import Request

from app.platform import rate_limit


def test_a_sent_key_does_not_change_who_the_caller_is() -> None:
    request = Request({"type": "http", "headers": [(b"x-api-key", b"made-up")], "client": ("198.51.100.7", 5000)})
    assert rate_limit.caller_of(request) == "ip:198.51.100.7"
