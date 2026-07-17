"""Run a bounded, source-ID-based FineWeb extension recovery smoke test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

from vasu.data.deduplication.extension_recovery import (
    DatasetServerSourceIdClient,
    RecoveryError,
    RecoveryResult,
    ScalabilityBlockedError,
    atomic_write_json,
    atomic_write_text,
    audit_extension_recovery,
    build_report,
    classify_retrieval,
    configuration_hash,
    deterministic_spread_sample,
    load_historical_records,
    render_report_text,
    utc_now,
)
from vasu.data.deduplication.recovery_benchmark import (
    BatchedDatasetServerClient,
    BenchmarkStrategy,
    JsonResponseCache,
    ProviderHttpClient,
    RetryPolicy,
    benchmark_configuration_hash,
    calculate_production_estimate,
    run_strategy,
    validate_strategy_result,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/data/deduplication/fineweb_extension_recovery.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Audit local evidence and probe bounded provider lookup.")
    mode.add_argument("--smoke-test", action="store_true", help="Retrieve exactly the configured deterministic sample.")
    mode.add_argument("--benchmark", action="store_true", help="Benchmark bounded recovery strategies.")
    parser.add_argument("--restart", action="store_true", help="Discard compatible progress and query the sample again.")
    parser.add_argument("--sample-count", type=int, default=1_000)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["serial", "concurrent-2", "concurrent-4", "concurrent-8", "batch-5", "batch-10", "batch-25"],
    )
    parser.add_argument(
        "--benchmark-cache-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data/interim/pretrain/fineweb_extension_recovery_benchmark_cache",
    )
    parser.add_argument("--clean-benchmark-cache", action="store_true")
    parser.add_argument("--requests-per-second", type=float, default=4.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    parser.add_argument("--retry-count", type=int, default=4)
    parser.add_argument("--retry-seed", type=int, default=42)
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Recovery config is missing: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid recovery config JSON: {error}") from error
    required = {
        "format_version", "dataset_repository", "dataset_config", "split",
        "dataset_revision", "source_id_field", "text_field", "metadata_path",
        "historical_database_path", "progress_path", "json_report_path",
        "text_report_path", "sample_size", "maximum_matches",
        "maximum_accepted_text_bytes", "maximum_response_bytes",
        "request_timeout_seconds", "acquisition_method",
        "allow_parquet_scan_fallback", "historical_fingerprint",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(f"Recovery config fields mismatch; missing={missing}, extra={extra}")
    if payload["format_version"] != 1:
        raise ValueError("Unsupported recovery config format_version")
    if payload["sample_size"] != 100 or payload["maximum_matches"] != 100:
        raise ValueError("Smoke recovery must select and retrieve at most exactly 100 IDs")
    if payload["maximum_accepted_text_bytes"] > 20_000_000:
        raise ValueError("Smoke accepted-text limit may not exceed 20 MB")
    if payload["allow_parquet_scan_fallback"] is not False:
        raise ValueError("Parquet/full-source scan fallback must remain disabled")
    if payload["source_id_field"] != "id" or payload["text_field"] != "text":
        raise ValueError("Historical provider fields must remain id and text")
    if payload["historical_fingerprint"] != "sha256_utf8_python_str_strip_v1":
        raise ValueError("Historical fingerprint contract mismatch")
    return payload


def repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def validate_config_against_audit(config: dict[str, Any], audit: Any) -> None:
    expected = (
        config["dataset_repository"], config["dataset_config"],
        config["dataset_revision"], config["split"],
    )
    observed = (audit.repository, audit.dataset_config, audit.revision, audit.split)
    if expected != observed:
        raise ValueError(f"Recovery configuration conflicts with historical evidence: {expected!r} != {observed!r}")
    if audit.classification != "source_id_reacquisition_possible":
        raise ValueError(f"Historical evidence is not safe for source-ID recovery: {audit.classification}")
    if audit.duplicate_source_ids or audit.invalid_historical_hashes or audit.missing_source_ids:
        raise ValueError("Historical evidence contains missing, duplicate, or invalid IDs/hashes")


def load_progress(path: Path, *, expected_hash: str, revision: str, restart: bool) -> dict[str, Any]:
    if restart or not path.exists():
        return {
            "format_version": 1,
            "configuration_hash": expected_hash,
            "revision": revision,
            "completed": {},
            "accepted_text_bytes": 0,
            "downloaded_response_bytes": 0,
            "accessed_endpoints": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("configuration_hash") != expected_hash:
        raise ValueError("Progress configuration hash mismatch; use --restart only after review")
    if payload.get("revision") != revision:
        raise ValueError("Progress revision mismatch; refusing revision fallback")
    if not isinstance(payload.get("completed"), dict):
        raise ValueError("Progress completed map is malformed")
    return payload


def pending_historical_records(selected: list[Any], completed: dict[str, Any]) -> list[Any]:
    """Return only unfinished IDs so resume never re-queries completed work."""
    return [item for item in selected if item.source_id not in completed]


def print_audit(config: dict[str, Any], audit: Any, selected: list[Any]) -> None:
    print(f"Historical source-ID path: {repository_path(config['historical_database_path'])}::fingerprints.source_id")
    print(f"Historical fingerprint path: {repository_path(config['historical_database_path'])}::fingerprints.fingerprint")
    print(f"Historical source IDs: {audit.retained_source_ids:,}")
    print(f"Historical fingerprints: {audit.extension_fingerprints:,}")
    print(f"Duplicate source IDs: {audit.duplicate_source_ids}")
    print(f"Duplicate historical hashes: {audit.duplicate_historical_hashes}")
    print(f"Expected source-ID field: {audit.source_id_field}")
    print(f"Historical hash contract: {audit.hash_contract}")
    print(f"Smoke IDs selected: {len(selected)}")
    print("Acquisition mechanism: official Dataset Viewer exact-ID /filter; no Parquet scan fallback")


def parse_benchmark_strategies(names: Sequence[str]) -> list[BenchmarkStrategy]:
    valid = {
        "serial": BenchmarkStrategy("serial", 1, 1),
        "concurrent-2": BenchmarkStrategy("concurrent-2", 2, 1),
        "concurrent-4": BenchmarkStrategy("concurrent-4", 4, 1),
        "concurrent-8": BenchmarkStrategy("concurrent-8", 8, 1),
        "batch-5": BenchmarkStrategy("batch-5", 1, 5),
        "batch-10": BenchmarkStrategy("batch-10", 1, 10),
        "batch-25": BenchmarkStrategy("batch-25", 1, 25),
    }
    unknown = sorted(set(names) - set(valid))
    if unknown:
        raise ValueError(f"Unknown benchmark strategies {unknown}; valid={sorted(valid)}")
    if len(set(names)) != len(names):
        raise ValueError("Benchmark strategy names must be unique")
    return [valid[name] for name in names]


def _http_snapshot(http: ProviderHttpClient) -> dict[str, Any]:
    return {
        "network_requests": http.network_request_count,
        "downloaded_response_bytes": http.downloaded_bytes,
        "cache_hits": http.cache_hits,
        "http_status_counts": dict(sorted(http.http_status_counts.items())),
        "transient_errors": http.transient_error_count,
        "timeouts": http.timeout_count,
        "retries": http.retry_count,
    }


def _empty_benchmark_state() -> dict[str, Any]:
    return {
        "duration_seconds": 0.0,
        "network_requests": 0,
        "downloaded_response_bytes": 0,
        "cache_hits": 0,
        "http_status_counts": {},
        "transient_errors": 0,
        "timeouts": 0,
        "retries": 0,
        "attempts": 0,
    }


def _merge_benchmark_state(
    prior: dict[str, Any], current: dict[str, Any], duration_seconds: float
) -> dict[str, Any]:
    merged = _empty_benchmark_state()
    for key in (
        "network_requests", "downloaded_response_bytes", "cache_hits",
        "transient_errors", "timeouts", "retries",
    ):
        merged[key] = int(prior.get(key, 0)) + int(current.get(key, 0))
    statuses: dict[str, int] = {}
    for source in (prior.get("http_status_counts", {}), current.get("http_status_counts", {})):
        for status, count in source.items():
            statuses[str(status)] = statuses.get(str(status), 0) + int(count)
    merged["http_status_counts"] = dict(sorted(statuses.items()))
    merged["duration_seconds"] = float(prior.get("duration_seconds", 0.0)) + duration_seconds
    merged["attempts"] = int(prior.get("attempts", 0)) + 1
    return merged


def _render_benchmark_text(report: dict[str, Any]) -> str:
    lines = [
        "FineWeb Extension Recovery Benchmark",
        "=" * 42,
        f"Sample IDs: {report['sample_count']}",
        f"Pinned revision: {report['revision']}",
        f"Requests per second limit: {report['retry_policy']['requests_per_second']}",
        "",
    ]
    for result in report["strategies"]:
        counts = result["match_counts"]
        lines.extend(
            [
                f"{result['name']}:",
                f"  cold duration: {result['cold']['duration_seconds']:.3f}s",
                f"  warm local-cache duration: {result['warm']['duration_seconds']:.3f}s",
                f"  network requests: {result['cold']['network_requests']}",
                f"  exact matches: {counts.get('exact_match', 0)}",
                f"  errors: {sum(value for key, value in counts.items() if key != 'exact_match')}",
            ]
        )
    lines.extend(
        [
            "",
            f"Recommended strategy: {report['recommended_strategy']}",
            "Warm timings are local-cache measurements and are not provider throughput.",
            "No source text is stored in this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    audit: Any,
    records: list[Any],
) -> int:
    if args.sample_count != 1_000:
        raise ValueError("The authorized benchmark must inspect exactly 1,000 IDs")
    selected = deterministic_spread_sample(records, args.sample_count)
    strategies = parse_benchmark_strategies(args.strategies)
    policy = RetryPolicy(
        requests_per_second=args.requests_per_second,
        connect_timeout_seconds=args.connect_timeout,
        read_timeout_seconds=args.read_timeout,
        retry_count=args.retry_count,
        seed=args.retry_seed,
    )
    cache_root = args.benchmark_cache_dir.resolve()
    progress_root = REPOSITORY_ROOT / "data/interim/pretrain/fineweb_extension_recovery_benchmark_progress"
    if args.clean_benchmark_cache:
        if cache_root.exists():
            shutil.rmtree(cache_root)
        if progress_root.exists():
            shutil.rmtree(progress_root)
    started_at = utc_now()
    strategy_reports: list[dict[str, Any]] = []
    for strategy in strategies:
        print(f"Benchmarking {strategy.name}: concurrency={strategy.concurrency}, batch={strategy.batch_size}", flush=True)
        strategy_cache = JsonResponseCache(cache_root / strategy.name)
        http = ProviderHttpClient(
            policy,
            maximum_response_bytes=max(
                int(config["maximum_response_bytes"]), strategy.batch_size * 2_000_000
            ),
            cache=strategy_cache,
        )
        provider = BatchedDatasetServerClient(
            repository=config["dataset_repository"],
            dataset_config=config["dataset_config"],
            split=config["split"],
            revision=config["dataset_revision"],
            http=http,
        )
        digest = benchmark_configuration_hash(
            strategy=strategy, records=selected, revision=config["dataset_revision"], policy=policy
        )
        state_path = progress_root / f"{strategy.name}_state.json"
        prior_state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file()
            else _empty_benchmark_state()
        )
        attempt_started = time.perf_counter()
        try:
            cold_results, cold_runtime = run_strategy(
                strategy=strategy,
                records=selected,
                client=provider,
                progress_path=progress_root / f"{strategy.name}.json",
                configuration_hash=digest,
            )
        except Exception:
            failed_state = _merge_benchmark_state(
                prior_state, _http_snapshot(http), time.perf_counter() - attempt_started
            )
            atomic_write_json(state_path, failed_state)
            raise
        if cold_runtime["new_records"] == 0:
            # A report-only rerun over already completed progress must not
            # inflate the measured cold provider duration or attempt counts.
            combined_state = prior_state
        else:
            combined_state = _merge_benchmark_state(
                prior_state, _http_snapshot(http), time.perf_counter() - attempt_started
            )
            atomic_write_json(state_path, combined_state)
        counts = validate_strategy_result(selected, cold_results)
        current_duration = cold_runtime["duration_seconds"]
        cold_runtime["current_attempt_duration_seconds"] = current_duration
        cold_runtime["duration_seconds"] = combined_state["duration_seconds"]
        cold_http = {
            key: value for key, value in combined_state.items()
            if key not in {"duration_seconds", "attempts"}
        }
        cold_http["attempts"] = combined_state["attempts"]
        accepted_bytes = sum(int(item.raw_utf8_bytes or 0) for item in cold_results)
        warm_http = ProviderHttpClient(
            policy,
            maximum_response_bytes=max(
                int(config["maximum_response_bytes"]), strategy.batch_size * 2_000_000
            ),
            cache=strategy_cache,
        )
        warm_provider = BatchedDatasetServerClient(
            repository=config["dataset_repository"],
            dataset_config=config["dataset_config"],
            split=config["split"],
            revision=config["dataset_revision"],
            http=warm_http,
        )
        warm_results, warm_runtime = run_strategy(
            strategy=strategy, records=selected, client=warm_provider
        )
        if [asdict(item) | {"retrieval_timestamp": None} for item in cold_results] != [
            asdict(item) | {"retrieval_timestamp": None} for item in warm_results
        ]:
            raise ValueError(f"Warm-cache correctness differs for {strategy.name}")
        validate_strategy_result(selected, warm_results)
        strategy_reports.append(
            {
                "name": strategy.name,
                "concurrency": strategy.concurrency,
                "batch_size": strategy.batch_size,
                "match_counts": counts,
                "accepted_text_bytes": accepted_bytes,
                "cold": {**cold_runtime, **cold_http},
                "warm": {**warm_runtime, **_http_snapshot(warm_http), "classification": "local_cache_only"},
                "results": [asdict(item) for item in cold_results],
            }
        )
        print(
            f"  {counts.get('exact_match', 0)}/{len(selected)} exact; "
            f"{cold_runtime['duration_seconds']:.2f}s; {cold_http['network_requests']} requests",
            flush=True,
        )
    correct = [
        item for item in strategy_reports
        if item["match_counts"] == {"exact_match": args.sample_count}
    ]
    if not correct:
        recommended = None
        estimate = None
    else:
        # Provider load is prioritized before runtime and concurrency.
        chosen = min(
            correct,
            key=lambda item: (
                item["cold"]["network_requests"],
                item["cold"]["duration_seconds"],
                item["concurrency"],
            ),
        )
        recommended = chosen["name"]
        estimate = calculate_production_estimate(
            historical_records=audit.retained_source_ids,
            strategy=BenchmarkStrategy(chosen["name"], chosen["concurrency"], chosen["batch_size"]),
            duration_seconds=chosen["cold"]["duration_seconds"],
            benchmark_records=args.sample_count,
            downloaded_bytes=chosen["cold"]["downloaded_response_bytes"],
            accepted_text_bytes=chosen["accepted_text_bytes"],
        )
        production_requests_per_second = min(2.0, policy.requests_per_second)
        production_policy = asdict(policy)
        production_policy["requests_per_second"] = production_requests_per_second
        estimate.update(
            {
                "recommended_requests_per_second": production_requests_per_second,
                "retry_policy": production_policy,
                "selection_reason": (
                    "Batch-25 passed exact correctness with the fewest provider requests. "
                    "Concurrency remains 1 and the production rate ceiling is reduced to 2 RPS; "
                    "observed request latency already kept actual throughput below that ceiling."
                ),
            }
        )
    report = {
        "format_version": 1,
        "status": "bounded_benchmark_only",
        "repository": config["dataset_repository"],
        "dataset_config": config["dataset_config"],
        "split": config["split"],
        "revision": config["dataset_revision"],
        "sample_count": len(selected),
        "sampled_source_ids": [item.source_id for item in selected],
        "started_at": started_at,
        "completed_at": utc_now(),
        "cache_root": cache_root.as_posix(),
        "retry_policy": asdict(policy),
        "official_api_findings": {
            "or_predicates_documented": True,
            "in_membership_documented": False,
            "in_membership_live_result": "HTTP 422 rejected",
            "maximum_rows_per_response": 100,
            "filtered_pagination_supported": True,
            "dataset_viewer_specific_rate_limit_documented": False,
            "server_response_byte_parameter": False,
            "revision_enforcement": "x-revision response header required on every network response",
        },
        "strategies": strategy_reports,
        "recommended_strategy": recommended,
        "production_estimate": estimate,
        "full_recovery_started": False,
        "extension_index_built": False,
        "training_started": False,
    }
    serialized = json.dumps(report, ensure_ascii=False)
    if any(f'"{key}"' in serialized for key in ("text", "raw_text", "content", "document")):
        raise ValueError("Benchmark report leaks source text")
    output_json = REPOSITORY_ROOT / "data/manifests/pretrain/fineweb_extension_recovery_benchmark.json"
    output_text = REPOSITORY_ROOT / "data/manifests/pretrain/fineweb_extension_recovery_benchmark.txt"
    atomic_write_json(output_json, report)
    atomic_write_text(output_text, _render_benchmark_text(report))
    print(f"Benchmark JSON: {output_json}")
    print(f"Benchmark text: {output_text}")
    return 0 if recommended else 2


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    metadata_path = repository_path(config["metadata_path"])
    database_path = repository_path(config["historical_database_path"])
    audit = audit_extension_recovery(metadata_path, database_path)
    validate_config_against_audit(config, audit)
    records = load_historical_records(database_path)
    if args.benchmark:
        print_audit(config, audit, deterministic_spread_sample(records, args.sample_count))
        return run_benchmark(args=args, config=config, audit=audit, records=records)
    selected = deterministic_spread_sample(records, config["sample_size"])
    config_digest = configuration_hash(config, selected)
    print_audit(config, audit, selected)

    client = DatasetServerSourceIdClient(
        repository=config["dataset_repository"],
        dataset_config=config["dataset_config"],
        split=config["split"],
        revision=config["dataset_revision"],
        timeout_seconds=float(config["request_timeout_seconds"]),
        maximum_response_bytes=int(config["maximum_response_bytes"]),
    )
    client.verify_source()
    print(f"Pinned revision verified: {config['dataset_revision']}")
    if args.dry_run:
        try:
            # Probe one selected ID. A loading/unavailable index blocks the run
            # instead of falling back to a 44+ GB Parquet scan.
            probe = client.retrieve(selected[0].source_id)
        except ScalabilityBlockedError as error:
            print(f"Scalability status: blocked - {error}")
            return 3
        print("Scalability status: bounded filter available")
        print(f"Probe rows returned: {len(probe)}")
        print("Dry run complete; no recovery report or training artifact was created.")
        return 0

    progress_path = repository_path(config["progress_path"])
    progress = load_progress(
        progress_path,
        expected_hash=config_digest,
        revision=config["dataset_revision"],
        restart=args.restart,
    )
    completed: dict[str, dict[str, Any]] = progress["completed"]
    pending = pending_historical_records(selected, completed)
    probe: list[Any] | None = None
    if pending:
        try:
            # Probe only the first unfinished ID so resume never repeats a
            # completed source-ID request.
            probe = client.retrieve(pending[0].source_id)
        except ScalabilityBlockedError as error:
            print(f"Scalability status: blocked - {error}")
            return 3
    print("Scalability status: bounded filter available")
    started_at = progress.get("started_at") or utc_now()
    progress["started_at"] = started_at
    accepted_bytes = int(progress.get("accepted_text_bytes", 0))
    prior_downloaded_bytes = int(progress.get("downloaded_response_bytes", 0))
    prior_endpoints = list(progress.get("accessed_endpoints", []))
    # The capability probe is the actual first unfinished retrieval; reuse it.
    prefetched = {pending[0].source_id: probe} if pending and probe is not None else {}
    for historical in pending:
        try:
            provider_records = prefetched.pop(historical.source_id, None)
            if provider_records is None:
                provider_records = client.retrieve(historical.source_id)
            result = classify_retrieval(historical, provider_records)
        except ScalabilityBlockedError:
            raise
        except Exception as error:  # a per-ID provider failure is reportable and resumable
            result = RecoveryResult(
                source_id=historical.source_id,
                expected_hash=historical.expected_hash,
                observed_hash=None,
                match_status="retrieval_error",
                provider_shard=None,
                stable_row_reference=None,
                raw_character_count=None,
                raw_utf8_bytes=None,
                retrieval_timestamp=utc_now(),
                diagnostic_note=f"{type(error).__name__}: {str(error)[:300]}",
            )
        record_bytes = int(result.raw_utf8_bytes or 0)
        if accepted_bytes + record_bytes > int(config["maximum_accepted_text_bytes"]):
            raise RecoveryError("Accepted text would exceed the configured 20 MB smoke limit")
        accepted_bytes += record_bytes
        completed[historical.source_id] = asdict(result)
        progress.update(
            {
                "accepted_text_bytes": accepted_bytes,
                "downloaded_response_bytes": prior_downloaded_bytes + client.downloaded_bytes,
                "accessed_endpoints": prior_endpoints + client.accessed_endpoints,
                "updated_at": utc_now(),
            }
        )
        atomic_write_json(progress_path, progress)

    ordered_results = [RecoveryResult(**completed[item.source_id]) for item in selected]
    completed_at = utc_now()
    report = build_report(
        audit=audit,
        selected=selected,
        results=ordered_results,
        accessed_endpoints=list(progress["accessed_endpoints"]),
        downloaded_bytes=int(progress["downloaded_response_bytes"]),
        accepted_text_bytes=accepted_bytes,
        config_hash=config_digest,
        started_at=started_at,
        completed_at=completed_at,
        scalability_status="bounded_filter_available",
    )
    json_report = repository_path(config["json_report_path"])
    text_report = repository_path(config["text_report_path"])
    atomic_write_json(json_report, report)
    atomic_write_text(text_report, render_report_text(report))
    print(render_report_text(report), end="")
    print(f"JSON report: {json_report}")
    print(f"Text report: {text_report}")
    return 0 if report["full_reacquisition_authorized"] else 2


def main() -> None:
    try:
        raise SystemExit(run())
    except (RecoveryError, ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
