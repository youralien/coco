"""One-off: write a training-ready copy of coco-records without degenerate rows.

Originals are never modified. For each session_* directory, the training files
(see py_utils.training_recorder) are copied to --out with these rows dropped:

* observations.jsonl  rows whose observer call failed or was rejected
                      (llm_metrics.success is False), hit the observer token
                      cap, or are repetition loops (compression ratio check
                      from sensing.observer_guard)
* tutor_calls.jsonl   rows whose tutor_output is a repetition loop (ratio only:
                      completion_tokens there is summed across agent turns, so
                      it is not comparable to a per-call cap)
* feedback.jsonl      reactions to a dropped observation
* decisions.jsonl     judge ticks whose fresh or history observations include a
                      dropped observation
* episodes.jsonl      nudges fired from a dropped decision
* observer_screenshots/  retained screenshots of dropped observations

Raw sensing state (actions.db, screenshots/, hotkey_captures/) is not training
data and is not copied. Kept screenshots are hard-linked (copied if the output
is on another volume), so the copy costs almost no extra disk.

    .venv/bin/python scripts/filter_training_records.py            # report only
    .venv/bin/python scripts/filter_training_records.py --write    # write --out
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "sensing"))

from sensing.observer_guard import degenerate_output_reason  # noqa: E402

# Matches sensing.segment_processor.OBSERVER_MAX_TOKENS (not imported: that
# module pulls in the whole sensing runtime).
OBSERVER_MAX_TOKENS = 4096
DEFAULT_RECORDS = Path("~/Library/Application Support/coco/coco-records")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a torn final line from a crash is not training data
    return rows


def _bad_observation(row: dict) -> bool:
    metrics = row.get("llm_metrics") or {}
    if metrics.get("success") is False:
        return True
    tokens = metrics.get("output_tokens") or metrics.get("completion_tokens") or 0
    return bool(
        degenerate_output_reason(
            row.get("observer_output") or "",
            output_tokens=int(tokens),
            max_tokens=OBSERVER_MAX_TOKENS,
        )
    )


def _bad_tutor(row: dict) -> bool:
    output = row.get("tutor_output") or ""
    return bool(degenerate_output_reason(output, output_tokens=0, max_tokens=0))


def filter_session(session: Path) -> tuple[dict[str, list[dict]], set[str], Counter]:
    """Return kept rows per file, dropped observation ids, and drop counts."""
    files = {
        name: _rows(session / name)
        for name in (
            "observations.jsonl",
            "tutor_calls.jsonl",
            "feedback.jsonl",
            "decisions.jsonl",
            "episodes.jsonl",
        )
    }
    dropped: Counter = Counter()
    bad_obs = {
        r["observation_id"]
        for r in files["observations.jsonl"]
        if r.get("observation_id") and _bad_observation(r)
    }
    bad_decisions = {
        r.get("decision_id")
        for r in files["decisions.jsonl"]
        if r.get("fresh_observation_id") in bad_obs
        or bad_obs.intersection(r.get("history_observation_ids") or [])
    }
    rules = {
        "observations.jsonl": lambda r: r.get("observation_id") in bad_obs,
        "tutor_calls.jsonl": _bad_tutor,
        "feedback.jsonl": lambda r: r.get("observation_id") in bad_obs,
        "decisions.jsonl": lambda r: r.get("decision_id") in bad_decisions,
        "episodes.jsonl": lambda r: r.get("decision_id") in bad_decisions,
    }
    kept: dict[str, list[dict]] = {}
    for name, rows in files.items():
        kept[name] = [r for r in rows if not rules[name](r)]
        dropped[name] = len(rows) - len(kept[name])
    return kept, bad_obs, dropped


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_session(session: Path, out: Path, kept: dict, bad_obs: set[str]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in kept.items():
        if (session / name).exists():
            with open(out / name, "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if (session / "manifest.json").exists():
        shutil.copy2(session / "manifest.json", out / "manifest.json")
    removed = 0
    shots = session / "observer_screenshots"
    if shots.is_dir():
        (out / "observer_screenshots").mkdir(exist_ok=True)
        for shot in shots.iterdir():
            # Retained files are named "<observation_id>_<i><suffix>".
            if shot.name.split("_", 1)[0] in bad_obs:
                removed += 1
            elif shot.is_file():
                _link_or_copy(shot, out / "observer_screenshots" / shot.name)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument(
        "--out", type=Path, help="output dir (default: <records>-filtered)"
    )
    parser.add_argument("--write", action="store_true", help="write the copy")
    args = parser.parse_args()
    records = args.records.expanduser()
    out_root = (args.out or records.with_name(records.name + "-filtered")).expanduser()
    if out_root.resolve() == records.resolve():
        sys.exit("--out must differ from --records; originals are never modified")
    sessions = sorted(p for p in records.glob("session_*") if p.is_dir())
    if not sessions:
        sys.exit(f"No session_* directories in {records}")
    if args.write and out_root.exists():
        sys.exit(f"{out_root} already exists; remove it or pass another --out")

    totals: Counter = Counter()
    print(f"Records: {records}")
    for session in sessions:
        kept, bad_obs, dropped = filter_session(session)
        shots = 0
        if args.write:
            shots = write_session(session, out_root / session.name, kept, bad_obs)
        else:
            shot_dir = session / "observer_screenshots"
            shots = sum(
                1
                for s in (shot_dir.iterdir() if shot_dir.is_dir() else [])
                if s.name.split("_", 1)[0] in bad_obs
            )
        dropped["observer_screenshots"] = shots
        totals.update(dropped)
        summary = ", ".join(
            f"{k.removesuffix('.jsonl')} -{v}" for k, v in dropped.items() if v
        )
        kept_obs = len(kept["observations.jsonl"])
        print(f"  {session.name}: {kept_obs} observations kept; {summary or 'clean'}")

    print(
        "\nDropped in total: "
        + (", ".join(f"{k} {v}" for k, v in totals.items() if v) or "nothing")
    )
    if args.write:
        print(f"Filtered copy written to {out_root}")
    else:
        print(f"Report only. Re-run with --write to create {out_root}")


if __name__ == "__main__":
    main()
