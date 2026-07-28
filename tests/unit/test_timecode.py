from captioner.shared.timecode import format_srt_timestamp


def test_format_srt_timestamp_uses_integer_milliseconds() -> None:
    assert format_srt_timestamp(3_726_045) == "01:02:06,045"
