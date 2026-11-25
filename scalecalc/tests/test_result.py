from scalecalc.chips import CHIPS
from scalecalc.result import NOTHING, Err, Ok, Option, Result, Some
from scalecalc.roofline import try_classify


def test_option_of_lifts_none_and_values():
    assert Option.of(None) is NOTHING
    assert Option.of(5) == Some(5)


def test_option_map_and_flat_map_short_circuit_on_nothing():
    assert Some(2).map(lambda x: x * 3) == Some(6)
    assert NOTHING.map(lambda x: x * 3) is NOTHING
    assert Some(2).flat_map(lambda x: Some(x + 1)) == Some(3)
    assert NOTHING.flat_map(lambda x: Some(x + 1)) is NOTHING


def test_option_get_or_else_and_unwrap():
    assert Some(9).get_or_else(0) == 9
    assert NOTHING.get_or_else(0) == 0
    assert Some(9).unwrap() == 9
    try:
        NOTHING.unwrap()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_result_try_captures_exception_as_err():
    ok = Result.try_(lambda: 10 / 2)
    err = Result.try_(lambda: 10 / 0)
    assert ok == Ok(5.0)
    assert err.is_err()
    assert isinstance(err.error, ZeroDivisionError)


def test_result_map_and_map_err():
    assert Ok(3).map(lambda x: x + 1) == Ok(4)
    assert Err("bad").map(lambda x: x + 1) == Err("bad")
    assert Err("bad").map_err(str.upper) == Err("BAD")
    assert Ok(3).map_err(str.upper) == Ok(3)


def test_try_classify_returns_err_for_invalid_bytes_moved():
    chip = CHIPS["tpu-v5e"]
    result = try_classify(chip, flops=1e9, bytes_moved=0)
    assert result.is_err()
    assert isinstance(result.error, ValueError)


def test_try_classify_returns_ok_for_valid_input():
    chip = CHIPS["tpu-v5e"]
    result = try_classify(chip, flops=1e9, bytes_moved=1e6)
    assert result.is_ok()
    assert result.unwrap().bound is not None
