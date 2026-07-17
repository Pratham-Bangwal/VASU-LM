"""Provider-friendly benchmark primitives for FineWeb source-ID recovery."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.parse

import requests

from .extension_recovery import (
    DATASET_SERVER_BASE_URL,
    HistoricalRecoveryRecord,
    RecoveryError,
    RecoveryResult,
    RetrievedSourceRecord,
    atomic_write_json,
    classify_retrieval,
)


TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class BenchmarkStrategy:
    name: str
    concurrency: int
    batch_size: int

    def __post_init__(self) -> None:
        if self.concurrency < 1 or self.batch_size < 1 or self.batch_size > 100:
            raise ValueError("Strategy concurrency and batch_size must be within bounds")


@dataclass(frozen=True)
class RetryPolicy:
    requests_per_second: float = 4.0
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 120.0
    retry_count: int = 4
    initial_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 60.0
    jitter_seconds: float = 0.25
    seed: int = 42

    def __post_init__(self) -> None:
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.connect_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise ValueError("connect/read timeouts must be positive")
        if self.retry_count < 0 or self.initial_backoff_seconds < 0:
            raise ValueError("retry policy values are invalid")
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("maximum backoff must be at least initial backoff")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds must be non-negative")


@dataclass(frozen=True)
class HttpResponseData:
    payload: dict[str, Any]
    headers: dict[str, str]
    byte_count: int
    from_cache: bool


class PermanentProviderError(RecoveryError):
    pass


class RetryExhaustedError(RecoveryError):
    pass


class ProviderTimeoutError(RecoveryError):
    pass


class RateLimiter:
    """Thread-safe fixed-interval limiter based on a monotonic clock."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = self.clock()
            delay = max(0.0, self._next_allowed - now)
            if delay:
                self.sleep(delay)
                now = self.clock()
            self._next_allowed = max(now, self._next_allowed) + self.interval


class JsonResponseCache:
    """Explicit temporary response cache; cached text never enters reports."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def clean(self) -> None:
        if self.directory.exists():
            shutil.rmtree(self.directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def read(self, key: str) -> HttpResponseData | None:
        path = self._path(key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return HttpResponseData(
            payload=payload["payload"],
            headers=payload["headers"],
            byte_count=int(payload["byte_count"]),
            from_cache=True,
        )

    def write(self, key: str, response: HttpResponseData) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self._path(key),
            {
                "payload": response.payload,
                "headers": response.headers,
                "byte_count": response.byte_count,
            },
        )


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        current = now or datetime.now(timezone.utc)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - current).total_seconds())


class ProviderHttpClient:
    """Rate-limited HTTP client with bounded retries and status accounting."""

    def __init__(
        self,
        policy: RetryPolicy,
        *,
        maximum_response_bytes: int,
        cache: JsonResponseCache | None = None,
        request: Callable[..., Any] = requests.get,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.policy = policy
        self.maximum_response_bytes = maximum_response_bytes
        self.cache = cache
        self.request = request
        self.sleep = sleep
        self.rate_limiter = RateLimiter(policy.requests_per_second, sleep=sleep)
        self._rng = random.Random(policy.seed)
        self._rng_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.http_status_counts: dict[str, int] = {}
        self.retry_count = 0
        self.transient_error_count = 0
        self.timeout_count = 0
        self.network_request_count = 0
        self.downloaded_bytes = 0
        self.cache_hits = 0

    def _jitter(self) -> float:
        with self._rng_lock:
            return self._rng.uniform(0.0, self.policy.jitter_seconds)

    def get_json(self, url: str) -> HttpResponseData:
        if self.cache:
            cached = self.cache.read(url)
            if cached is not None:
                with self._stats_lock:
                    self.cache_hits += 1
                return cached
        for attempt in range(self.policy.retry_count + 1):
            self.rate_limiter.acquire()
            try:
                response = self.request(
                    url,
                    headers={"User-Agent": "VASU-extension-recovery-benchmark/1.0"},
                    timeout=(
                        self.policy.connect_timeout_seconds,
                        self.policy.read_timeout_seconds,
                    ),
                )
            except requests.Timeout as error:
                with self._stats_lock:
                    self.timeout_count += 1
                    self.transient_error_count += 1
                    self.network_request_count += 1
                if attempt >= self.policy.retry_count:
                    raise ProviderTimeoutError("Provider request timed out") from error
                self._sleep_before_retry(attempt, None)
                continue
            except requests.ConnectionError as error:
                with self._stats_lock:
                    self.transient_error_count += 1
                    self.network_request_count += 1
                if attempt >= self.policy.retry_count:
                    raise RetryExhaustedError("Provider connection retries exhausted") from error
                self._sleep_before_retry(attempt, None)
                continue
            status = int(response.status_code)
            body = bytes(response.content)
            with self._stats_lock:
                self.network_request_count += 1
                self.http_status_counts[str(status)] = self.http_status_counts.get(str(status), 0) + 1
                self.downloaded_bytes += len(body)
            if len(body) > self.maximum_response_bytes:
                raise RecoveryError(
                    f"Provider response exceeded {self.maximum_response_bytes} bytes"
                )
            if status in TRANSIENT_HTTP_STATUSES:
                with self._stats_lock:
                    self.transient_error_count += 1
                if attempt >= self.policy.retry_count:
                    raise RetryExhaustedError(f"HTTP {status} retries exhausted")
                self._sleep_before_retry(attempt, response.headers.get("Retry-After"))
                continue
            if 400 <= status < 500:
                raise PermanentProviderError(f"Permanent provider HTTP {status}: {body[:300]!r}")
            if status >= 500:
                raise RetryExhaustedError(f"Unexpected provider HTTP {status}")
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError) as error:
                raise RecoveryError("Provider returned invalid JSON") from error
            if not isinstance(payload, dict):
                raise RecoveryError("Provider returned non-object JSON")
            result = HttpResponseData(
                payload=payload,
                headers={str(key): str(value) for key, value in response.headers.items()},
                byte_count=len(body),
                from_cache=False,
            )
            if self.cache:
                self.cache.write(url, result)
            return result
        raise AssertionError("unreachable")

    def _sleep_before_retry(self, attempt: int, retry_after: str | None) -> None:
        delay = parse_retry_after(retry_after)
        if delay is None:
            delay = min(
                self.policy.maximum_backoff_seconds,
                self.policy.initial_backoff_seconds * (2**attempt),
            ) + self._jitter()
        with self._stats_lock:
            self.retry_count += 1
        self.sleep(delay)


class BatchedDatasetServerClient:
    """Exact source-ID lookup using documented OR filter predicates."""

    def __init__(
        self,
        *,
        repository: str,
        dataset_config: str,
        split: str,
        revision: str,
        http: ProviderHttpClient,
    ) -> None:
        self.repository = repository
        self.dataset_config = dataset_config
        self.split = split
        self.revision = revision
        self.http = http

    def retrieve_many(self, source_ids: Sequence[str]) -> dict[str, list[RetrievedSourceRecord]]:
        if not source_ids or len(source_ids) > 100:
            raise ValueError("A filter batch must contain 1..100 IDs")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("A filter batch contains duplicate requested IDs")
        predicates = []
        for source_id in source_ids:
            escaped = source_id.replace("'", "''")
            predicates.append(f'"id"=\'{escaped}\'')
        where = " OR ".join(predicates)
        parameters = {
            "dataset": self.repository,
            "config": self.dataset_config,
            "split": self.split,
            "where": where,
            "offset": 0,
            "length": len(source_ids),
        }
        url = f"{DATASET_SERVER_BASE_URL}/filter?{urllib.parse.urlencode(parameters)}"
        response = self.http.get_json(url)
        served_revision = response.headers.get("x-revision") or response.headers.get("X-Revision")
        if served_revision != self.revision:
            raise RecoveryError(
                f"Dataset Viewer revision mismatch: expected {self.revision}, got {served_revision!r}"
            )
        rows = response.payload.get("rows")
        total = response.payload.get("num_rows_total")
        if not isinstance(rows, list) or not isinstance(total, int) or total < len(rows):
            raise RecoveryError("Malformed Dataset Viewer filter response")
        requested = set(source_ids)
        grouped: dict[str, list[RetrievedSourceRecord]] = {source_id: [] for source_id in source_ids}
        for item in rows:
            if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
                raise RecoveryError("Malformed provider row")
            row = item["row"]
            source_id = row.get("id")
            if not isinstance(source_id, str) or source_id not in requested:
                raise RecoveryError(f"Provider returned an unrequested source ID: {source_id!r}")
            grouped[source_id].append(
                RetrievedSourceRecord(
                    source_id=source_id,
                    text=row.get("text") if isinstance(row.get("text"), str) else "",
                    provider_shard=row.get("file_path") if isinstance(row.get("file_path"), str) else None,
                    stable_row_reference=(
                        f"dataset-viewer-row:{item['row_idx']}"
                        if isinstance(item.get("row_idx"), int)
                        else None
                    ),
                    endpoint=url,
                    revision=self.revision,
                )
            )
        # If total exceeds returned rows, at least one requested ID is duplicated.
        # Fetching additional text would add provider load; disqualify the batch.
        if total > len(rows):
            raise RecoveryError(
                f"Filtered result has {total} matches for {len(source_ids)} IDs; duplicate provider ID suspected"
            )
        return grouped


def chunk_records(
    records: Sequence[HistoricalRecoveryRecord], batch_size: int
) -> list[list[HistoricalRecoveryRecord]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [list(records[index : index + batch_size]) for index in range(0, len(records), batch_size)]


def run_strategy(
    *,
    strategy: BenchmarkStrategy,
    records: Sequence[HistoricalRecoveryRecord],
    client: BatchedDatasetServerClient,
    progress_path: Path | None = None,
    configuration_hash: str | None = None,
) -> tuple[list[RecoveryResult], dict[str, Any]]:
    """Run one strategy while preserving deterministic result order."""
    started = time.perf_counter()
    completed: dict[str, dict[str, Any]] = {}
    if progress_path and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("configuration_hash") != configuration_hash:
            raise ValueError("Benchmark progress configuration hash mismatch")
        completed = dict(progress.get("completed", {}))
    pending = [record for record in records if record.source_id not in completed]
    batches = chunk_records(pending, strategy.batch_size)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def process(batch: list[HistoricalRecoveryRecord]) -> list[RecoveryResult]:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            grouped = client.retrieve_many([item.source_id for item in batch])
            return [classify_retrieval(item, grouped[item.source_id]) for item in batch]
        finally:
            with lock:
                active -= 1

    # Submit only the configured number of batches. This bounds both active and
    # queued provider work and lets an error stop without draining a 1,000-item
    # pre-submitted queue whose results were never committed.
    with ThreadPoolExecutor(max_workers=strategy.concurrency) as executor:
        batch_iterator = iter(batches)
        futures: dict[Future[list[RecoveryResult]], list[HistoricalRecoveryRecord]] = {}
        for _ in range(strategy.concurrency):
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break
            futures[executor.submit(process, batch)] = batch
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                results = future.result()
                with lock:
                    for result in results:
                        completed[result.source_id] = asdict(result)
                    if progress_path:
                        atomic_write_json(
                            progress_path,
                            {
                                "format_version": 1,
                                "configuration_hash": configuration_hash,
                                "strategy": asdict(strategy),
                                "completed": completed,
                            },
                        )
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    continue
                futures[executor.submit(process, batch)] = batch
    ordered = [RecoveryResult(**completed[item.source_id]) for item in records]
    elapsed = time.perf_counter() - started
    return ordered, {
        "duration_seconds": elapsed,
        "maximum_in_flight": maximum_active,
        "resumed_records": len(records) - len(pending),
        "new_records": len(pending),
    }


def validate_strategy_result(
    records: Sequence[HistoricalRecoveryRecord], results: Sequence[RecoveryResult]
) -> dict[str, int]:
    if [item.source_id for item in results] != [item.source_id for item in records]:
        raise ValueError("Strategy output ordering differs from deterministic sample ordering")
    counts: dict[str, int] = {}
    for result in results:
        counts[result.match_status] = counts.get(result.match_status, 0) + 1
        if "text" in asdict(result):
            raise ValueError("Source text leaked into benchmark result")
    if len(results) != len(records):
        raise ValueError("Strategy did not produce one result per sampled ID")
    return counts


def benchmark_configuration_hash(
    *,
    strategy: BenchmarkStrategy,
    records: Sequence[HistoricalRecoveryRecord],
    revision: str,
    policy: RetryPolicy,
) -> str:
    payload = {
        "strategy": asdict(strategy),
        "source_ids": [item.source_id for item in records],
        "revision": revision,
        "policy": asdict(policy),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def calculate_production_estimate(
    *,
    historical_records: int,
    strategy: BenchmarkStrategy,
    duration_seconds: float,
    benchmark_records: int,
    downloaded_bytes: int,
    accepted_text_bytes: int,
) -> dict[str, Any]:
    if historical_records < 1 or benchmark_records < 1 or duration_seconds <= 0:
        raise ValueError("Production estimate inputs must be positive")
    request_count = math.ceil(historical_records / strategy.batch_size)
    scale = historical_records / benchmark_records
    return {
        "full_request_count": request_count,
        "estimated_runtime_seconds": duration_seconds * scale,
        "estimated_downloaded_response_bytes": round(downloaded_bytes * scale),
        "estimated_accepted_text_bytes": round(accepted_text_bytes * scale),
        "estimated_temporary_storage_bytes": round(accepted_text_bytes * scale * 2),
        "recommended_checkpoint_interval_records": 1_000,
        "recommended_batch_size": strategy.batch_size,
        "recommended_concurrency": strategy.concurrency,
    }
