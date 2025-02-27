import interpreto


def test_get_version():
    assert interpreto.get_version() == '0.1.0.dev0'
