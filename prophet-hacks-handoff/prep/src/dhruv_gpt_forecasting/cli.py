"""Small command-line wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arena_agent import forecast_arena_event, forecast_arena_payload_for_ensemble
from .arena_batch import main as arena_batch_main
from .arena_eval import main as arena_eval_main
from .kalshi_auth import kalshi_credential_status
from .kalshi_public import main as kalshi_events_main
from .live_readiness import main as live_readiness_main
from .preflight import main as preflight_main
from .prophet_api import forecast_scores, health, prophet_api_status, write_events
from .vendor_evidence import main as vendor_evidence_main


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("arena-eval")
    sub.add_parser("arena-batch")
    sub.add_parser("credentials")
    sub.add_parser("kalshi-events")
    sub.add_parser("live-readiness")
    sub.add_parser("preflight")
    sub.add_parser("runbook")
    sub.add_parser("vendor-evidence")
    prophet_events = sub.add_parser("prophet-events")
    prophet_events.add_argument("--status", choices=["all", "open", "closed"], default="open")
    prophet_events.add_argument("-o", "--output", type=Path, default=Path("runs/dhruv_gemini/prophet_events.json"))
    sub.add_parser("prophet-scores")
    sub.add_parser("prophet-health")
    predict = sub.add_parser("predict-json")
    predict.add_argument("event_json", type=Path)
    predict.add_argument("--live-data", action="store_true")
    predict.add_argument("--no-gpt", action="store_true")
    predict.add_argument("--mode", default="real_live_smoke")
    arena_predict = sub.add_parser("predict-arena-json")
    arena_predict.add_argument("event_json", type=Path)
    arena_predict.add_argument("--live-data", action="store_true")
    arena_predict.add_argument("--no-gpt", action="store_true")
    arena_predict.add_argument("--mode", default="real_live_smoke")
    prophet_predict = sub.add_parser("predict-prophet-json")
    prophet_predict.add_argument("event_json", type=Path)
    prophet_predict.add_argument("--live-data", action="store_true")
    prophet_predict.add_argument("--no-gpt", action="store_true")
    args, rest = parser.parse_known_args()
    if args.cmd == "arena-eval":
        import sys

        sys.argv = [sys.argv[0], *rest]
        return arena_eval_main()
    if args.cmd == "arena-batch":
        import sys

        sys.argv = [sys.argv[0], *rest]
        return arena_batch_main()
    if args.cmd == "runbook":
        import sys

        sys.argv = [sys.argv[0], "runbook", *rest]
        return arena_batch_main()
    if args.cmd == "kalshi-events":
        import sys

        sys.argv = [sys.argv[0], *rest]
        return kalshi_events_main()
    if args.cmd == "live-readiness":
        import sys

        sys.argv = [sys.argv[0], *rest]
        return live_readiness_main()
    if args.cmd == "preflight":
        import sys

        sys.argv = [sys.argv[0], *rest]
        return preflight_main()
    if args.cmd == "vendor-evidence":
        import sys

        sys.argv = [sys.argv[0], *rest]
        return vendor_evidence_main()
    if args.cmd == "credentials":
        print(json.dumps({
            "kalshi": kalshi_credential_status(),
            "prophet_arena": prophet_api_status(),
        }, indent=2, sort_keys=True))
        return 0
    if args.cmd == "prophet-health":
        print(json.dumps(health(), indent=2, sort_keys=True))
        return 0
    if args.cmd == "prophet-events":
        events = write_events(args.output, status=args.status)
        print(json.dumps({"output": str(args.output), "n_events": len(events)}, indent=2, sort_keys=True))
        return 0
    if args.cmd == "prophet-scores":
        print(json.dumps(forecast_scores(), indent=2, sort_keys=True))
        return 0
    event = json.loads(args.event_json.read_text(encoding="utf-8"))
    if args.cmd in {"predict-json", "predict-arena-json"}:
        response = forecast_arena_payload_for_ensemble(
            event,
            use_gpt=False if args.no_gpt else None,
            use_live_data=True if args.live_data else None,
            mode=args.mode,
        )
        print(json.dumps(response, indent=2, sort_keys=True))
        return 0
    if args.cmd == "predict-prophet-json":
        if isinstance(event, list):
            event = event[0]
        decision = forecast_arena_event(
            event,
            use_gpt=False if args.no_gpt else None,
            use_live_data=True if args.live_data else None,
        )
        print(json.dumps(decision.to_prediction_response(), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
