from slmbench.models.base import _extract_json


def test_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_json_in_markdown_fence():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json(text) == {"a": 1}


def test_json_with_preamble_text():
    text = 'Here is the extracted data:\n{"a": 1}\nLet me know if you need more.'
    assert _extract_json(text) == {"a": 1}


def test_invalid_json_returns_none():
    assert _extract_json("not json at all") is None
