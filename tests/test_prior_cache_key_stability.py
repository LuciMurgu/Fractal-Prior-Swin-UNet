from fractal_swin_unet.fractal.cache_disk import make_cache_key


def test_cache_key_stability() -> None:
    params_a = {"lfd.window": 15, "lfd.box_sizes": [2, 4, 8], "lfd.stride": 4}
    params_b = {"lfd.box_sizes": [2, 4, 8], "lfd.stride": 4, "lfd.window": 15}

    key_a = make_cache_key("sample", "dbc_lfd", params_a)
    key_b = make_cache_key("sample", "dbc_lfd", params_b)

    assert key_a == key_b

    key_c = make_cache_key("sample", "dbc_lfd", {"lfd.window": 17})
    assert key_a != key_c
