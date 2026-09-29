import pytest

from src.common import (
    CHINESE_VOICES,
    hotword_prompt,
    is_allowed_audio_type,
    parse_hotwords,
    request_id,
)


def test_hotwords_are_trimmed_and_deduplicated() -> None:
    assert parse_hotwords('[" 元气森林 ", "东方树叶", "元气森林"]') == ["元气森林", "东方树叶"]


def test_hotwords_require_json_array() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        parse_hotwords("元气森林")


def test_hotword_prompt_uses_official_vocabulary_form() -> None:
    assert hotword_prompt(["元气森林", "Coca-Cola"]) == "Vocabulary: 元气森林, Coca-Cola."
    assert hotword_prompt([]) is None


def test_request_id_is_generated_and_bounded() -> None:
    assert request_id()
    assert len(request_id("x" * 200)) == 128


def test_voice_contract() -> None:
    assert CHINESE_VOICES == ("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric")


def test_browser_audio_mime_with_codec_is_allowed() -> None:
    assert is_allowed_audio_type("audio/webm;codecs=opus")
    assert not is_allowed_audio_type("text/plain")
