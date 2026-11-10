from voice_agent.text import SentenceStreamer, clean_for_speech, detect_language, split_sentences, stream_sentences


def test_hindi_danda_and_english_full_stop():
    text = "दिल्ली भारत की राजधानी है। It is a big city. क्या आप वहाँ गए हैं?"
    assert split_sentences(text) == ["दिल्ली भारत की राजधानी है।", "It is a big city.", "क्या आप वहाँ गए हैं?"]


def test_sentence_emitted_as_soon_as_boundary_seen():
    s = SentenceStreamer()
    assert list(s.feed("नमस्ते, मैं ठीक हूँ")) == []
    assert list(s.feed("।")) == []  # can't know yet whether more follows the ender
    assert list(s.feed(" आप")) == ["नमस्ते, मैं ठीक हूँ।"]
    assert list(s.feed(" कैसे हैं?")) == []
    assert list(s.flush()) == ["आप कैसे हैं?"]


def test_token_by_token_streaming_matches_whole_text():
    text = "The answer is 1081. That is 23 times 47! Anything else?"
    tokens = [text[i : i + 3] for i in range(0, len(text), 3)]
    assert list(stream_sentences(tokens)) == split_sentences(text)


def test_decimals_and_abbreviations_do_not_split():
    assert split_sentences("Pi is about 3.14 today. Dr. Rao agrees.") == ["Pi is about 3.14 today.", "Dr. Rao agrees."]


def test_short_pieces_merge_after_first():
    out = split_sentences("Hi. Ok. Sure. This is a longer sentence.", min_chars=12)
    assert out[0] == "Hi."  # eager first sentence gets the voice going
    assert out[1] == "Ok. Sure. This is a longer sentence."


def test_long_run_on_sentence_is_split_at_comma():
    long = "यह एक बहुत लंबा वाक्य है, " * 10
    out = split_sentences(long, max_chars=80)
    assert len(out) > 1 and all(len(p) <= 100 for p in out)


def test_newline_is_a_boundary_and_markdown_is_removed():
    assert split_sentences("**Answer:** 42\n- next item") == ["Answer: 42", "next item"]


def test_clean_for_speech_strips_emoji_and_links():
    assert clean_for_speech("Done 😀 see [the site](http://x.y) ✅") == "Done see the site"


def test_detect_language():
    assert detect_language("आज मंगलवार है।") == "hi"
    assert detect_language("Today is Tuesday.") == "en"
    assert detect_language("मेरा flight कब है?") == "hi"
    assert detect_language("1930") == "en"
