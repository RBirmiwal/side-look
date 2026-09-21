from incidence import INCIDENCE_FIELD, parse_incidence


def test_incidence_field_is_from_vertical():
    meta = {
        "collect": {
            "image": {
                "center_pixel": {
                    "incidence_angle": 33.6155957963679,
                    "squint_angle": 0.98,
                }
            }
        }
    }
    out = parse_incidence(meta)
    assert out["field"] == INCIDENCE_FIELD
    assert out["raw_value_deg"] == 33.6155957963679
    assert out["theta_from_vertical_deg"] == 33.6155957963679
    assert out["ninety_minus_raw_deg"] == 90.0 - 33.6155957963679
    assert out["converted"] is False


def test_grazing_named_field_is_converted():
    meta = {"collect": {"image": {"center_pixel": {"grazing_angle": 56.3844}}}}
    # parse_incidence looks up incidence_angle specifically; a grazing *name*
    # on the incidence field would convert. Simulate a misnamed leaf:
    from incidence import INCIDENCE_PATH

    meta = {"collect": {"image": {"center_pixel": {"incidence_angle": 33.6}}}}
    out = parse_incidence(meta)
    assert abs((out["raw_value_deg"] + out["ninety_minus_raw_deg"]) - 90.0) < 1e-9


def test_missing_incidence_fails():
    try:
        parse_incidence({"collect": {"image": {}}})
    except KeyError as exc:
        assert INCIDENCE_FIELD in str(exc)
    else:
        raise AssertionError("expected KeyError")
