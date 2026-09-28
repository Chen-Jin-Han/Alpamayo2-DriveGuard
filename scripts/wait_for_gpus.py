#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


def gpu_rows() -> list[dict[str, int]]:
    query = "index,memory.free,utilization.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True
    )
    rows = []
    for line in output.splitlines():
        index, free, util = [int(value.strip()) for value in line.split(",")]
        rows.append({"index": index, "free": free, "util": util})
    return rows


def busy_gpu_indices() -> set[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
    ).strip()
    if not output:
        return set()
    uuid_to_index = {}
    mapping = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    for line in mapping.splitlines():
        index, uuid = [value.strip() for value in line.split(",", 1)]
        uuid_to_index[uuid] = int(index)
    return {uuid_to_index[uuid.strip()] for uuid in output.splitlines() if uuid.strip() in uuid_to_index}


def selected_are_idle(selected: list[int], min_free_mib: int, max_util: int) -> bool:
    busy = busy_gpu_indices()
    rows = {row["index"]: row for row in gpu_rows()}
    return all(
        index not in busy
        and rows[index]["free"] >= min_free_mib
        and rows[index]["util"] <= max_util
        for index in selected
    )


def continuously_idle(
    selected: list[int], settle_seconds: int, min_free_mib: int, max_util: int
) -> bool:
    deadline = time.monotonic() + settle_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(5.0, remaining))
        if not selected_are_idle(selected, min_free_mib, max_util):
            return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait politely for genuinely idle GPUs, then run a command")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--guard-count",
        type=int,
        default=None,
        help="require this many GPUs to remain idle, while exposing only --count to the child",
    )
    parser.add_argument("--min-free-mib", type=int, default=28000)
    parser.add_argument("--max-util", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--settle-seconds", type=int, default=5)
    parser.add_argument("--lock", default="/tmp/alpamayo2-driveguard-gpu.lock")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    guard_count = args.count if args.guard_count is None else args.guard_count
    if guard_count < args.count:
        parser.error("--guard-count must be at least --count")
    Path(args.lock).parent.mkdir(parents=True, exist_ok=True)
    with open(args.lock, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            busy = busy_gpu_indices()
            candidates = [
                row for row in gpu_rows()
                if row["index"] not in busy
                and row["free"] >= args.min_free_mib
                and row["util"] <= args.max_util
            ]
            if len(candidates) >= guard_count:
                selected = [row["index"] for row in candidates[: args.count]]
                guarded = [row["index"] for row in candidates[:guard_count]]
                print(
                    f"Verifying continuous idle window: selected={selected}, guarded={guarded}, "
                    f"seconds={args.settle_seconds}",
                    flush=True,
                )
                if not continuously_idle(
                    guarded, args.settle_seconds, args.min_free_mib, args.max_util
                ):
                    print(
                        f"Yielding: GPU state changed during idle window; guarded={guarded}",
                        flush=True,
                    )
                    time.sleep(args.poll_seconds)
                    continue
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))
                print(
                    f"GPUs continuously idle; launching on physical GPU(s) {selected}",
                    flush=True,
                )
                return subprocess.call(command, env=env)
            print(
                f"Waiting: need {guard_count} idle GPU(s) ({args.count} for child), "
                f"candidates={candidates}, busy={sorted(busy)}",
                flush=True,
            )
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
