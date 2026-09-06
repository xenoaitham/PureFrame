import json

from pureframe.utils.timing import PhaseTimers


def test_phase_accumulates_and_counts():
    timers = PhaseTimers()
    with timers.phase("a"):
        pass
    with timers.phase("a"):
        pass
    with timers.phase("b"):
        pass

    d = timers.as_dict()
    assert d["a"]["calls"] == 2
    assert d["b"]["calls"] == 1
    assert 0 <= d["a"]["seconds"] < 1


def test_phase_records_on_exception():
    timers = PhaseTimers()
    try:
        with timers.phase("boom"):
            raise ValueError("x")
    except ValueError:
        pass
    assert timers.as_dict()["boom"]["calls"] == 1


def test_machine_line_roundtrip():
    timers = PhaseTimers()
    with timers.phase("render"):
        pass
    assert "render" in json.dumps(timers.as_dict())
    assert "Phase timings" in timers.summary()
