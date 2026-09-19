import csv

import pytest

from loadtest.check_results import Summary, Thresholds, evaluate, main, read_aggregated

HEADER = [
    "Type", "Name", "Request Count", "Failure Count", "Median Response Time", "Average Response Time",
    "Min Response Time", "Max Response Time", "Average Content Size", "Requests/s", "Failures/s",
    "50%", "66%", "75%", "80%", "90%", "95%", "98%", "99%", "99.9%", "99.99%", "100%",
]


def write_stats(path, requests=6000, failures=0, p99=120, rps=50.0, aggregated=True):
    rows = [["POST", "/predict", requests, failures, 20, 25, 2, 300, 100, rps, 0] + [p99] * 11]
    if aggregated:
        rows.append(["", "Aggregated", requests, failures, 20, 25, 2, 300, 100, rps, 0] + [p99] * 11)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return path


T = Thresholds(target_rps=50, max_p99_ms=250, max_error_rate=0.001, min_rps_ratio=0.9)


def checks(**kw):
    return {c.name: c.passed for c in evaluate(Summary(**{"requests": 6000, "failures": 0, "p99_ms": 100, "rps": 50, **kw}), T)}


def test_all_within_limits_passes():
    assert all(checks().values())


@pytest.mark.parametrize(
    "kw,failing",
    [
        ({"p99_ms": 251}, "p99 latency"),
        ({"failures": 7}, "error rate"),  # 7/6000 = 0.117%
        ({"rps": 44.9}, "throughput"),
    ],
)
def test_each_threshold_trips_independently(kw, failing):
    result = checks(**kw)
    assert not result[failing]
    assert sum(not v for v in result.values()) == 1


def test_boundaries_are_inclusive():
    assert all(checks(p99_ms=250, failures=6, rps=45.0).values())


def test_reads_aggregated_row(tmp_path):
    s = read_aggregated(write_stats(tmp_path / "s.csv", requests=1000, failures=2, p99=180, rps=49.5))
    assert s == Summary(requests=1000, failures=2, p99_ms=180, rps=49.5)
    assert s.error_rate == pytest.approx(0.002)


def test_main_exit_codes(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    ok = write_stats(tmp_path / "ok.csv")
    slow = write_stats(tmp_path / "slow.csv", p99=900)
    assert main([str(ok), "--target-rps", "50"]) == 0
    assert "performance gate PASSED" in capsys.readouterr().out
    assert main([str(slow), "--target-rps", "50"]) == 1
    assert "| p99 latency | FAIL | 900 ms | <= 250 ms |" in capsys.readouterr().out
    assert main([str(ok), "--target-rps", "100"]) == 1  # 50 rps achieved < 90 required


def test_thresholds_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MAX_P99_MS", "100")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main([str(write_stats(tmp_path / "s.csv", p99=120))]) == 1


def test_unusable_results_exit_2(tmp_path):
    assert main([str(tmp_path / "missing.csv")]) == 2
    assert main([str(write_stats(tmp_path / "noagg.csv", aggregated=False))]) == 2
    assert main([str(write_stats(tmp_path / "empty.csv", requests=0))]) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("Name,Request Count\nAggregated,lots\n")
    assert main([str(bad)]) == 2


def test_writes_github_step_summary(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    main([str(write_stats(tmp_path / "s.csv")), "--target-rps", "50"])
    assert "### Performance gate PASSED" in summary.read_text()
