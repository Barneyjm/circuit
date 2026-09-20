"""Telling a media state apart from data that happens to mention media."""

import pytest

from s1proto.media import split_media_state


@pytest.mark.parametrize(
    "state",
    [
        {"image": None},  # found by the fuzzer: a null under the word "image"
        {"image": "cover.jpg", "id": 7},  # a record about an image
        {"image": 3, "0": "x"},
        {"audio": {"nested": 1}},
        {"image": "a red dog on a beach"},  # a caption, not a file
        {"image": "x.png", "audio": 5},  # only one of them is a spec
    ],
)
def test_data_that_merely_mentions_media_is_scored_as_json(state):
    assert split_media_state(state) is None


@pytest.mark.parametrize(
    "state, modality",
    [
        ({"image": "https://example.com/a.png"}, "vision"),
        ({"image": "data:image/png;base64,AAAA"}, "vision"),
        ({"audio": "https://example.com/a.wav"}, "audio"),
        ({"audio": "call.wav", "text": "Tuesday 9:14"}, "audio"),
        ({"image": b"\x89PNG"}, "vision"),
    ],
)
def test_real_media_states_still_split(state, modality):
    out = split_media_state(state)
    assert out is not None and out[1] == modality


def test_a_spec_with_stray_keys_is_still_an_error():
    """A real media state with junk beside it is malformed, and saying so beats
    silently scoring a data URI as text."""
    with pytest.raises(ValueError, match="unexpected keys"):
        split_media_state({"image": "https://example.com/a.png", "caption": "hi"})


def test_both_media_keys_is_an_error_when_both_are_specs():
    with pytest.raises(ValueError, match="one of image or audio"):
        split_media_state({"image": "a.png", "audio": "b.wav"})
