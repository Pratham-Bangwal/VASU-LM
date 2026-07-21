from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import threading
import time
from typing import Any

import pytest
import requests

from scripts.reacquire_fineweb_extension import parse_args, parse_benchmark_strategies
from vasu.data.deduplication.extension_recovery import (
    HistoricalRecoveryRecord,
    RecoveryError,
    RetrievedSourceRecord,
    atomic_write_json,
    deterministic_spread_sample,
    historical_extension_fingerprint,
)
from vasu.data.deduplication.recovery_benchmark import (
    BatchedDatasetServerClient,
    BenchmarkStrategy,
    JsonResponseCache,
    PermanentProviderError,
    ProviderHttpClient,
    ProviderTimeoutError,
    RateLimiter,
    RetryPolicy,
    benchmark_configuration_hash,
    calculate_production_estimate,
    parse_retry_after,
    run_strategy,
    validate_strategy_result,
)


REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status
        self._payload = payload or {}
        self.content = json.dumps(self._payload).encode()
        self.headers = headers or {"x-revision": REVISION}

    def json(self) -> dict[str, Any]:
        return self._payload


def records(count: int) -> list[HistoricalRecoveryRecord]:
    return [
        HistoricalRecoveryRecord(f"id-{index:05}", historical_extension_fingerprint(f"text-{index}"))
        for index in range(count)
    ]


class FakeBatchClient:
    def __init__(self, *, delay: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.delay = delay

    def retrieve_many(self, source_ids: list[str]) -> dict[str, list[RetrievedSourceRecord]]:
        self.calls.append(tuple(source_ids))
        if self.delay:
            time.sleep((int(source_ids[0].split("-")[-1]) % 3) * 0.001)
        return {
            source_id: [
                RetrievedSourceRecord(
                    source_id,
                    f"text-{int(source_id.split('-')[-1])}",
                    "shard",
                    source_id,
                    "endpoint",
                    REVISION,
                )
            ]
            for source_id in source_ids
        }


def no_wait_policy(**changes: Any) -> RetryPolicy:
    values = {
        "requests_per_second": 1_000_000.0,
        "retry_count": 2,
        "initial_backoff_seconds": 0.0,
        "maximum_backoff_seconds": 0.0,
        "jitter_seconds": 0.0,
    }
    values.update(changes)
    return RetryPolicy(**values)


def test_deterministic_benchmark_id_sample() -> None:
    source = records(2_001)
    assert deterministic_spread_sample(source, 1_000) == deterministic_spread_sample(reversed(source), 1_000)


def test_concurrency_preserves_output_order() -> None:
    source = records(40)
    results, _ = run_strategy(
        strategy=BenchmarkStrategy("concurrent-8", 8, 1),
        records=source,
        client=FakeBatchClient(delay=True),  # type: ignore[arg-type]
    )
    assert [item.source_id for item in results] == [item.source_id for item in source]


def test_seeded_jitter_does_not_mutate_global_random_state() -> None:
    random.seed(9283)
    before = random.getstate()
    client = ProviderHttpClient(no_wait_policy(jitter_seconds=1.0), maximum_response_bytes=100, sleep=lambda _: None)
    client._jitter()
    assert random.getstate() == before


def test_rate_limiter_behavior() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2.0, clock=lambda: now[0], sleep=sleep)
    limiter.acquire()
    limiter.acquire()
    assert sleeps == [0.5]


def test_retry_after_seconds_and_http_date() -> None:
    assert parse_retry_after("7") == 7.0
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_retry_after("Thu, 01 Jan 2026 00:00:09 GMT", now=now) == 9.0


def test_retry_after_is_honored() -> None:
    responses = [
        FakeResponse(429, headers={"Retry-After": "3"}),
        FakeResponse(200, {"ok": True}),
    ]
    sleeps: list[float] = []
    client = ProviderHttpClient(
        no_wait_policy(), maximum_response_bytes=1_000,
        request=lambda *args, **kwargs: responses.pop(0), sleep=sleeps.append,
    )
    assert client.get_json("https://example.test").payload == {"ok": True}
    assert 3.0 in sleeps and client.retry_count == 1


def test_transient_retry() -> None:
    responses = [FakeResponse(503), FakeResponse(200, {"ok": True})]
    client = ProviderHttpClient(
        no_wait_policy(), maximum_response_bytes=1_000,
        request=lambda *args, **kwargs: responses.pop(0), sleep=lambda _: None,
    )
    client.get_json("https://example.test")
    assert client.transient_error_count == 1 and client.retry_count == 1


def test_permanent_error_is_not_retried() -> None:
    client = ProviderHttpClient(
        no_wait_policy(), maximum_response_bytes=1_000,
        request=lambda *args, **kwargs: FakeResponse(422), sleep=lambda _: None,
    )
    with pytest.raises(PermanentProviderError):
        client.get_json("https://example.test")
    assert client.retry_count == 0


def test_dataset_viewer_invalid_query_422_is_retried() -> None:
    responses = [
        FakeResponse(422, {"error": "A query parameter is invalid"}),
        FakeResponse(200, {"ok": True}),
    ]
    client = ProviderHttpClient(
        no_wait_policy(), maximum_response_bytes=1_000,
        request=lambda *args, **kwargs: responses.pop(0), sleep=lambda _: None,
    )
    assert client.get_json("https://example.test").payload == {"ok": True}
    assert client.retry_count == 1
    assert client.transient_error_count == 1


def test_timeout_classification() -> None:
    def request(*args: object, **kwargs: object) -> Any:
        raise requests.Timeout("timeout")

    client = ProviderHttpClient(
        no_wait_policy(retry_count=0), maximum_response_bytes=1_000,
        request=request, sleep=lambda _: None,
    )
    with pytest.raises(ProviderTimeoutError):
        client.get_json("https://example.test")
    assert client.timeout_count == 1


def test_duplicate_result_detection() -> None:
    source = records(1)
    bad = [
        RetrievedSourceRecord(source[0].source_id, "text-0", None, None, "e", REVISION),
        RetrievedSourceRecord(source[0].source_id, "text-0", None, None, "e", REVISION),
    ]
    from vasu.data.deduplication.extension_recovery import classify_retrieval

    assert classify_retrieval(source[0], bad).match_status == "duplicate_source_id_in_provider"


def test_exact_hash_verification_under_concurrency() -> None:
    source = records(100)
    results, _ = run_strategy(
        strategy=BenchmarkStrategy("concurrent-4", 4, 1),
        records=source,
        client=FakeBatchClient(),  # type: ignore[arg-type]
    )
    assert validate_strategy_result(source, results) == {"exact_match": 100}


def test_resume_after_partial_concurrent_completion(tmp_path: Path) -> None:
    source = records(5)
    strategy = BenchmarkStrategy("concurrent-2", 2, 1)
    digest = benchmark_configuration_hash(
        strategy=strategy, records=source, revision=REVISION, policy=no_wait_policy()
    )
    first_result, _ = run_strategy(
        strategy=BenchmarkStrategy("one", 1, 1), records=source[:2],
        client=FakeBatchClient(),  # type: ignore[arg-type]
    )
    progress = tmp_path / "progress.json"
    atomic_write_json(
        progress,
        {"configuration_hash": digest, "completed": {item.source_id: asdict(item) for item in first_result}},
    )
    client = FakeBatchClient()
    result, metadata = run_strategy(
        strategy=strategy, records=source, client=client,  # type: ignore[arg-type]
        progress_path=progress, configuration_hash=digest,
    )
    assert len(result) == 5 and metadata["resumed_records"] == 2


def test_completed_id_is_not_fetched_twice_on_resume(tmp_path: Path) -> None:
    source = records(3)
    strategy = BenchmarkStrategy("serial", 1, 1)
    digest = benchmark_configuration_hash(
        strategy=strategy, records=source, revision=REVISION, policy=no_wait_policy()
    )
    initial, _ = run_strategy(
        strategy=strategy, records=source[:1], client=FakeBatchClient(),  # type: ignore[arg-type]
    )
    progress = tmp_path / "progress.json"
    atomic_write_json(progress, {"configuration_hash": digest, "completed": {source[0].source_id: asdict(initial[0])}})
    client = FakeBatchClient()
    run_strategy(
        strategy=strategy, records=source, client=client,  # type: ignore[arg-type]
        progress_path=progress, configuration_hash=digest,
    )
    assert all(source[0].source_id not in call for call in client.calls)


def test_in_flight_requests_are_bounded() -> None:
    _, metadata = run_strategy(
        strategy=BenchmarkStrategy("concurrent-4", 4, 1),
        records=records(30), client=FakeBatchClient(delay=True),  # type: ignore[arg-type]
    )
    assert metadata["maximum_in_flight"] <= 4


def test_provider_failure_does_not_drain_presubmitted_queue() -> None:
    class FailingClient(FakeBatchClient):
        def retrieve_many(self, source_ids: list[str]) -> dict[str, list[RetrievedSourceRecord]]:
            self.calls.append(tuple(source_ids))
            if source_ids == ["id-00000"]:
                raise RecoveryError("stop")
            time.sleep(0.01)
            source_id = source_ids[0]
            return {
                source_id: [
                    RetrievedSourceRecord(
                        source_id, f"text-{int(source_id.split('-')[-1])}",
                        "shard", source_id, "endpoint", REVISION,
                    )
                ]
            }

    client = FailingClient()
    with pytest.raises(RecoveryError, match="stop"):
        run_strategy(
            strategy=BenchmarkStrategy("concurrent-4", 4, 1),
            records=records(100), client=client,  # type: ignore[arg-type]
        )
    assert len(client.calls) <= 4


def test_benchmark_cache_separation(tmp_path: Path) -> None:
    one = JsonResponseCache(tmp_path / "one")
    two = JsonResponseCache(tmp_path / "two")
    response = FakeResponse(200, {"value": 1})
    client = ProviderHttpClient(
        no_wait_policy(), maximum_response_bytes=1_000,
        cache=one, request=lambda *args, **kwargs: response, sleep=lambda _: None,
    )
    client.get_json("url")
    assert one.read("url") is not None and two.read("url") is None


def test_cold_and_warm_cache_reporting(tmp_path: Path) -> None:
    calls = [0]

    def request(*args: object, **kwargs: object) -> FakeResponse:
        calls[0] += 1
        return FakeResponse(200, {"ok": True})

    cache = JsonResponseCache(tmp_path / "cache")
    cold = ProviderHttpClient(no_wait_policy(), maximum_response_bytes=1_000, cache=cache, request=request, sleep=lambda _: None)
    assert cold.get_json("url").from_cache is False
    warm = ProviderHttpClient(no_wait_policy(), maximum_response_bytes=1_000, cache=cache, request=request, sleep=lambda _: None)
    assert warm.get_json("url").from_cache is True
    assert calls[0] == 1 and warm.cache_hits == 1


def test_strategy_result_validation_rejects_wrong_order() -> None:
    source = records(2)
    result, _ = run_strategy(
        strategy=BenchmarkStrategy("serial", 1, 1), records=source,
        client=FakeBatchClient(),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="ordering"):
        validate_strategy_result(source, list(reversed(result)))


def test_source_text_is_excluded_from_results() -> None:
    source = records(1)
    result, _ = run_strategy(
        strategy=BenchmarkStrategy("serial", 1, 1), records=source,
        client=FakeBatchClient(),  # type: ignore[arg-type]
    )
    assert "text" not in asdict(result[0])


def test_batched_client_enforces_pinned_revision() -> None:
    payload = {"num_rows_total": 0, "rows": []}
    http = ProviderHttpClient(
        no_wait_policy(), maximum_response_bytes=1_000,
        request=lambda *args, **kwargs: FakeResponse(200, payload, {"x-revision": "wrong"}),
        sleep=lambda _: None,
    )
    client = BatchedDatasetServerClient(
        repository="r", dataset_config="c", split="train", revision=REVISION, http=http
    )
    with pytest.raises(RecoveryError, match="revision mismatch"):
        client.retrieve_many(["id"])


def test_production_estimate_calculation() -> None:
    estimate = calculate_production_estimate(
        historical_records=379_247,
        strategy=BenchmarkStrategy("batch-25", 1, 25),
        duration_seconds=10.0,
        benchmark_records=1_000,
        downloaded_bytes=1_000_000,
        accepted_text_bytes=500_000,
    )
    assert estimate["full_request_count"] == 15_170
    assert estimate["estimated_runtime_seconds"] == pytest.approx(3_792.47)


def test_existing_smoke_cli_behavior_remains_available() -> None:
    args = parse_args(["--smoke-test"])
    assert args.smoke_test is True and args.benchmark is False and args.sample_count == 1_000
    assert parse_benchmark_strategies(["serial"])[0] == BenchmarkStrategy("serial", 1, 1)
