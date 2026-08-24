from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze_results
from .anthropic_batch import run_anthropic_smoke, step_anthropic_batch
from .donation_positive_control import create_positive_control_manifest, step_positive_control
from .commitment_order import create_commitment_manifest, step_commitment_order
from .task_load import create_task_load_manifest, step_task_load
from .valence_factorial import create_valence_manifest, step_valence_factorial
from .openai_controls import step_openai_positive, step_openai_valence
from .opaque_clause_control import create_opaque_manifest, step_opaque_control, analyze_opaque_results
from .moral_inversion import create_inversion_manifest, step_inversion, analyze_inversion_results
from .manifest import create_manifest
from .runner import run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Within-trajectory value-leakage experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Create a frozen randomized manifest")
    manifest.add_argument("--config", type=Path, required=True)
    manifest.add_argument("--out", type=Path, required=True)

    run = subparsers.add_parser("run", help="Run or resume a manifest")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--concurrency", type=int)
    run.add_argument("--limit", type=int, help="Run only the first N pending trials (smoke testing)")

    analyze = subparsers.add_parser("analyze", help="Parse and summarize raw JSONL")
    analyze.add_argument("--results", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)

    anthropic_smoke = subparsers.add_parser(
        "anthropic-smoke", help="Run one synchronous two-turn Anthropic request-shape smoke"
    )
    anthropic_smoke.add_argument("--manifest", type=Path, required=True)
    anthropic_smoke.add_argument("--out", type=Path, required=True)

    anthropic_batch = subparsers.add_parser(
        "anthropic-batch-step", help="Submit, poll, or advance the resumable two-phase Anthropic batch"
    )
    anthropic_batch.add_argument("--manifest", type=Path, required=True)
    anthropic_batch.add_argument("--out", type=Path, required=True)

    positive_manifest = subparsers.add_parser(
        "positive-control-manifest", help="Freeze the original Donation Bet baseline batch"
    )
    positive_manifest.add_argument("--config", type=Path, required=True)
    positive_manifest.add_argument("--out", type=Path, required=True)

    positive_step = subparsers.add_parser(
        "positive-control-step", help="Submit, poll, or advance the original Donation Bet positive control"
    )
    positive_step.add_argument("--manifest", type=Path, required=True)
    positive_step.add_argument("--out", type=Path, required=True)

    commitment_manifest = subparsers.add_parser(
        "commitment-manifest", help="Freeze the randomized pre/post commitment-order experiment"
    )
    commitment_manifest.add_argument("--config", type=Path, required=True)
    commitment_manifest.add_argument("--out", type=Path, required=True)

    commitment_step = subparsers.add_parser(
        "commitment-batch-step", help="Submit, poll, or advance the commitment-order batches"
    )
    commitment_step.add_argument("--manifest", type=Path, required=True)
    commitment_step.add_argument("--out", type=Path, required=True)

    task_load_manifest = subparsers.add_parser(
        "task-load-manifest", help="Freeze the randomized value-by-neutral-task-load experiment"
    )
    task_load_manifest.add_argument("--config", type=Path, required=True)
    task_load_manifest.add_argument("--out", type=Path, required=True)

    task_load_step = subparsers.add_parser(
        "task-load-batch-step", help="Submit, poll, advance, and analyze the task-load batches"
    )
    task_load_step.add_argument("--manifest", type=Path, required=True)
    task_load_step.add_argument("--out", type=Path, required=True)

    valence_manifest = subparsers.add_parser(
        "valence-manifest", help="Freeze the semantic-consequence by explicit-label factorial"
    )
    valence_manifest.add_argument("--config", type=Path, required=True)
    valence_manifest.add_argument("--out", type=Path, required=True)

    valence_step = subparsers.add_parser(
        "valence-batch-step", help="Submit, poll, and analyze the valence factorial batch"
    )
    valence_step.add_argument("--manifest", type=Path, required=True)
    valence_step.add_argument("--out", type=Path, required=True)

    openai_positive = subparsers.add_parser(
        "openai-positive-step", help="Submit, poll, and analyze the OpenAI Batch positive control"
    )
    openai_positive.add_argument("--manifest", type=Path, required=True)
    openai_positive.add_argument("--out", type=Path, required=True)

    openai_valence = subparsers.add_parser(
        "openai-valence-step", help="Submit, poll, and analyze the OpenAI Batch valence factorial"
    )
    openai_valence.add_argument("--manifest", type=Path, required=True)
    openai_valence.add_argument("--out", type=Path, required=True)

    opaque_manifest = subparsers.add_parser("opaque-manifest", help="Freeze opaque-identifier by clause-order control")
    opaque_manifest.add_argument("--config", type=Path, required=True)
    opaque_manifest.add_argument("--out", type=Path, required=True)
    opaque_step = subparsers.add_parser("opaque-batch-step", help="Submit, poll, and analyze opaque-clause batch")
    opaque_step.add_argument("--manifest", type=Path, required=True)
    opaque_step.add_argument("--out", type=Path, required=True)
    opaque_analyze = subparsers.add_parser("opaque-analyze", help="Analyze a downloaded opaque-clause batch JSONL offline")
    opaque_analyze.add_argument("--manifest", type=Path, required=True)
    opaque_analyze.add_argument("--results", type=Path, required=True)
    opaque_analyze.add_argument("--out", type=Path, required=True)
    inversion_manifest = subparsers.add_parser("inversion-manifest", help="Freeze congruent-versus-inverted moral-label control")
    inversion_manifest.add_argument("--config", type=Path, required=True)
    inversion_manifest.add_argument("--out", type=Path, required=True)
    inversion_step = subparsers.add_parser("inversion-batch-step", help="Submit, poll, and analyze moral-inversion batch")
    inversion_step.add_argument("--manifest", type=Path, required=True)
    inversion_step.add_argument("--out", type=Path, required=True)
    inversion_analyze = subparsers.add_parser("inversion-analyze", help="Analyze downloaded moral-inversion JSONL offline")
    inversion_analyze.add_argument("--manifest", type=Path, required=True)
    inversion_analyze.add_argument("--results", type=Path, required=True)
    inversion_analyze.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "manifest":
        manifest = create_manifest(args.config, args.out)
        print(f"Wrote {len(manifest['trials'])} trials to {args.out}")
        print(f"Manifest core SHA-256: {manifest['manifest_core_sha256']}")
    elif args.command == "run":
        receipt = run_manifest(args.manifest, args.out, args.concurrency, args.limit)
        print(f"Records total: {receipt['records_total']}")
        print(f"Status counts: {receipt['status_counts']}")
    elif args.command == "analyze":
        summary = analyze_results(args.results, args.out)
        print(f"Records analyzed: {summary['records_total']}")
        print(f"Primary contrast: {summary['primary_directional_contrast']['difference']}")
    elif args.command == "anthropic-smoke":
        record = run_anthropic_smoke(args.manifest, args.out)
        print(f"Smoke trial: {record['trial_id']} {record['arm_id']} {record['status']}")
    elif args.command == "anthropic-batch-step":
        result = step_anthropic_batch(args.manifest, args.out)
        print(f"Batch action: {result['action']}")
        if result.get("processing_status"):
            print(f"Processing status: {result['processing_status']}")
        if result.get("request_counts"):
            print(f"Request counts: {result['request_counts']}")
    elif args.command == "positive-control-manifest":
        manifest = create_positive_control_manifest(args.config, args.out)
        print(f"Wrote {len(manifest['baseline_trials'])} baseline trials to {args.out}")
        print(f"Manifest core SHA-256: {manifest['manifest_core_sha256']}")
    elif args.command == "positive-control-step":
        result = step_positive_control(args.manifest, args.out)
        print(f"Positive-control action: {result['action']}")
        if result.get("processing_status"):
            print(f"Processing status: {result['processing_status']}")
        if result.get("request_counts"):
            print(f"Request counts: {result['request_counts']}")
        if result.get("threshold") is not None:
            print(f"Frozen threshold: {result['threshold']}")
    elif args.command == "commitment-manifest":
        manifest = create_commitment_manifest(args.config, args.out)
        print(f"Wrote {len(manifest['trials'])} trials to {args.out}")
        print(f"Manifest core SHA-256: {manifest['manifest_core_sha256']}")
    elif args.command == "commitment-batch-step":
        result = step_commitment_order(args.manifest, args.out)
        print(f"Commitment-order action: {result['action']}")
        if result.get("processing_status"):
            print(f"Processing status: {result['processing_status']}")
        if result.get("request_counts"):
            print(f"Request counts: {result['request_counts']}")
    elif args.command == "task-load-manifest":
        manifest = create_task_load_manifest(args.config, args.out)
        print(f"Wrote {len(manifest['trials'])} trials to {args.out}")
        print(f"Manifest core SHA-256: {manifest['manifest_core_sha256']}")
    elif args.command == "task-load-batch-step":
        result = step_task_load(args.manifest, args.out)
        print(f"Task-load action: {result['action']}")
        if result.get("processing_status"):
            print(f"Processing status: {result['processing_status']}")
        if result.get("request_counts"):
            print(f"Request counts: {result['request_counts']}")
    elif args.command == "valence-manifest":
        manifest = create_valence_manifest(args.config, args.out)
        print(f"Wrote {len(manifest['trials'])} trials to {args.out}")
        print(f"Manifest core SHA-256: {manifest['manifest_core_sha256']}")
    elif args.command == "valence-batch-step":
        result = step_valence_factorial(args.manifest, args.out)
        print(f"Valence-factorial action: {result['action']}")
        if result.get("processing_status"):
            print(f"Processing status: {result['processing_status']}")
        if result.get("request_counts"):
            print(f"Request counts: {result['request_counts']}")
    elif args.command == "openai-positive-step":
        result = step_openai_positive(args.manifest, args.out)
        print(f"OpenAI positive-control action: {result['action']}")
        if result.get("status"):
            print(f"Batch status: {result['status']}")
        if result.get("request_counts"):
            print(f"Request counts: {result['request_counts']}")
        if result.get("threshold") is not None:
            print(f"Frozen threshold: {result['threshold']}")
    elif args.command == "openai-valence-step":
        result = step_openai_valence(args.manifest, args.out)
        print(f"OpenAI valence-factorial action: {result['action']}")
        if result.get("status"):
            print(f"Batch status: {result['status']}")
        if result.get("request_counts"):
            print(f"Request counts: {result['request_counts']}")
    elif args.command == "opaque-manifest":
        manifest = create_opaque_manifest(args.config, args.out)
        print(f"Wrote {len(manifest['trials'])} trials to {args.out}")
        print(f"Manifest core SHA-256: {manifest['manifest_core_sha256']}")
    elif args.command == "opaque-batch-step":
        result = step_opaque_control(args.manifest, args.out)
        print(f"Opaque-clause action: {result['action']}")
        if result.get("batch_id"):
            print(f"Batch ID: {result['batch_id']}")
        if result.get("processing_status"):
            print(f"Processing status: {result['processing_status']}")
        if result.get("request_counts"):
            print(f"Request counts: {result['request_counts']}")
    elif args.command == "opaque-analyze":
        summary = analyze_opaque_results(args.manifest, args.results, args.out)
        print(f"Records analyzed: {summary['records_total']}")
        print(f"Valid estimates: {summary['valid_estimates']}")
    elif args.command == "inversion-manifest":
        manifest = create_inversion_manifest(args.config, args.out)
        print(f"Wrote {len(manifest['trials'])} trials to {args.out}")
        print(f"Manifest core SHA-256: {manifest['manifest_core_sha256']}")
    elif args.command == "inversion-batch-step":
        result = step_inversion(args.manifest, args.out)
        print(f"Moral-inversion action: {result['action']}")
        if result.get("batch_id"):
            print(f"Batch ID: {result['batch_id']}")
        if result.get("processing_status"):
            print(f"Processing status: {result['processing_status']}")
        if result.get("request_counts"):
            print(f"Request counts: {result['request_counts']}")
    elif args.command == "inversion-analyze":
        summary = analyze_inversion_results(args.manifest, args.results, args.out)
        print(f"Records analyzed: {summary['records_total']}")
        print(f"Valid estimates: {summary['valid_estimates']}")
