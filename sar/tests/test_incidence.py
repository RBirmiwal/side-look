from incidence import INCIDENCE_FIELD, parse_incidence


def test_incidence_field_is_from_vertical():
    meta = {"collect": {"image": {"center_pixel": {"incidence_angle": 33.6155957963679, "squint_angle": 0.98}}}}
    out = parse_incidence(meta)
    assert out["field"] == INCIDENCE_FIELD
    assert out["raw_value_deg"] == 33.6155957963679
    assert out["theta_from_vertical_deg"] == 33.6155957963679
    assert out["converted"] is False


def test_missing_incidence_fails():
    try:
        parse_incidence({"collect": {"image": {}}})
    except KeyError as exc:
        assert INCIDENCE_FIELD in str(exc)
    else:
        raise AssertionError("expected KeyError")
