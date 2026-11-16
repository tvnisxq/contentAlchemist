from media_worker.media.downloader import validate_youtube_url


def test_valid_youtube_url():
    assert validate_youtube_url(
        "https://www.youtube.com/watch?v=abc123"
    )


def test_valid_short_url():
    assert validate_youtube_url(
        "https://youtu.be/abc123"
    )


def test_reject_non_youtube_url():
    assert not validate_youtube_url(
        "https://example.com/video"
    )


def test_reject_http_url():
    assert not validate_youtube_url(
        "http://youtube.com/watch?v=abc123"
    )


if __name__ == "__main__":
    test_valid_youtube_url()
    test_valid_short_url()
    test_reject_non_youtube_url()
    test_reject_http_url()

    print("All downloader validation tests passed!")
