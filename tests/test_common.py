import pytest

from src.common import (
    CHINESE_VOICES,
    hotword_prompt,
    is_allowed_audio_type,
    parse_hotwords,
    request_id,
    split_tts_sentences,
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


def test_split_tts_sentences_preserves_punctuation_and_newlines() -> None:
    assert split_tts_sentences("第一句。第二句！\n第三句？") == [
        "第一句。",
        "第二句！",
        "第三句？",
    ]


def test_split_tts_sentences_keeps_decimal_and_splits_english_period() -> None:
    assert split_tts_sentences("价格是3.14元。 Version 2.0 works. Next sentence.") == [
        "价格是3.14元。",
        "Version 2.0 works.",
        "Next sentence.",
    ]


def test_split_tts_sentences_prefers_secondary_punctuation_for_long_text() -> None:
    text = "一二三四五六，七八九十一二三四五六七八九十"
    assert split_tts_sentences(text, max_chars=10) == [
        "一二三四五六，",
        "七八九十一二三四五六",
        "七八九十",
    ]
