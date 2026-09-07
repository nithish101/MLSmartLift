from app.features import ModelInput
from app.inference_log import InferenceLogger
from app.metrics import INFERENCE_LOG_DROPPED

FEATURES = ModelInput(exercise="squat", weight=225, rir=2, bodyweight=180)


class FakeSession:
    def __init__(self, sink, fail=False):
        self.sink, self.fail = sink, fail

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, rows):
        if self.fail:
            raise RuntimeError("db down")
        self.sink.extend(rows)

    def commit(self):
        pass


def dropped():
    return INFERENCE_LOG_DROPPED._value.get()


def test_flush_writes_buffered_rows_in_batches():
    sink = []
    log = InferenceLogger(lambda: FakeSession(sink), flush_seconds=60, buffer_size=2000)
    for _ in range(1200):
        log.record(FEATURES, 3, 6.5, False, "predict")
    assert log.flush() == 1200
    assert len(sink) == 1200
    assert sink[0]["model_version"] == 3 and sink[0]["weight"] == 225 and sink[0]["source"] == "predict"
    assert log.flush() == 0


def test_full_buffer_drops_and_counts():
    log = InferenceLogger(lambda: FakeSession([]), flush_seconds=60, buffer_size=2)
    before = dropped()
    for _ in range(5):
        log.record(FEATURES, 1, 6.0, True, "predict")
    assert dropped() - before == 3


def test_failed_flush_drops_batch_and_keeps_running():
    log = InferenceLogger(lambda: FakeSession([], fail=True), flush_seconds=60, buffer_size=10)
    log.record(FEATURES, 1, 6.0, False, "predict")
    before = dropped()
    assert log.flush() == 0
    assert dropped() - before == 1


def test_background_thread_flushes_and_stop_drains():
    sink = []
    log = InferenceLogger(lambda: FakeSession(sink), flush_seconds=0.05, buffer_size=10)
    log.start()
    log.record(FEATURES, 1, 6.0, False, "predict")
    import time

    deadline = time.monotonic() + 5
    while not sink and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(sink) == 1
    log.record(FEATURES, 1, 6.0, False, "predict")
    log.stop()
    assert len(sink) == 2
