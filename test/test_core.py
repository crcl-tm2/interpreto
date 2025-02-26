import interpreto


def test_get_version():
    assert interpreto.get_version() == "unknown"
