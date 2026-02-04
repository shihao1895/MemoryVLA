import os
import json
import re
import glob
from typing import Dict, Any, List, Optional

TASKS = ["SG", "IM", "RC3", "RC5", "RC9"]


def parse_iter_k(step_dir_name: str) -> str:
    m = re.search(r"step-(\d+)", step_dir_name)
    if not m:
        return step_dir_name
    step = int(m.group(1))
    if step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def iter_k_to_int(iter_k: str) -> int:
    if iter_k.endswith("k"):
        try:
            return int(iter_k[:-1]) * 1000
        except ValueError:
            return 0
    try:
        return int(iter_k)
    except ValueError:
        return 0


def read_json_objects_stream(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        s = f.read()

    dec = json.JSONDecoder()
    i, n = 0, len(s)
    out: List[Dict[str, Any]] = []

    while True:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break

        obj, j = dec.raw_decode(s, i)
        if isinstance(obj, dict):
            out.append(obj)
        i = j

    return out


def parse_one_file(path: str, tasks: List[str]) -> Dict[str, Any]:
    rows = read_json_objects_stream(path)

    task_success: Dict[str, float] = {}
    avg_success: Optional[float] = None

    for r in rows:
        if "task_success" in r:
            ts = r.get("task_success", {})
            if isinstance(ts, dict):
                for t in tasks:
                    if t in ts and isinstance(ts[t], (int, float)):
                        task_success[t] = float(ts[t])
            if "avg_success" in r and isinstance(r["avg_success"], (int, float)):
                avg_success = float(r["avg_success"])
            continue

        if "task" in r and "success" in r:
            t = r["task"]
            s = r["success"]
            if t in tasks and isinstance(s, (int, float)):
                task_success[t] = float(s)

    covered = sum(1 for t in tasks if t in task_success)
    missing = [t for t in tasks if t not in task_success]
    is_complete = (covered == len(tasks))

    if avg_success is None and covered > 0:
        avg_success = sum(task_success[t] for t in tasks if t in task_success) / covered

    return {
        "path": path,
        "task_success": task_success,
        "avg_success": avg_success,  # None or float
        "covered": covered,
        "is_complete": is_complete,
        "missing": missing,
    }


def pick_best_candidate(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cands:
        return None

    complete = [c for c in cands if c["is_complete"] and c["avg_success"] is not None]
    if complete:
        return max(complete, key=lambda c: c["avg_success"])

    def score(c):
        avg = c["avg_success"]
        return (c["covered"], -1.0 if avg is None else float(avg))

    return max(cands, key=score)


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x * 100:5.1f}"


def extract_and_print_mikasa(
    root_paths: List[str],
    dir_flag: str = "eval_mikasa",
    print_style: str = "md",  # 'md' or 'text'
):
    headers = ["iter", "AVG"] + TASKS

    header_fmt = "{:>6s}  {:>6s}  {:>6s}  {:>6s}  {:>6s}  {:>6s}  {:>6s}"
    row_fmt = "{iter:>6s}  {AVG:>6s}  {SG:>6s}  {IM:>6s}  {RC3:>6s}  {RC5:>6s}  {RC9:>6s}"

    for root in root_paths:
        eval_dir = os.path.join(root, dir_flag)
        if not os.path.isdir(eval_dir):
            print(f"No eval dir: {eval_dir}\n")
            continue

        step_dirs = sorted(
            d for d in os.listdir(eval_dir)
            if os.path.isdir(os.path.join(eval_dir, d))
        )

        table_rows = []
        incomplete_report = []

        for step_dir in step_dirs:
            step_path = os.path.join(eval_dir, step_dir)
            iter_k = parse_iter_k(step_dir)

            paths = sorted(glob.glob(os.path.join(step_path, "*.jsonl")))
            cands = []
            for p in paths:
                try:
                    cands.append(parse_one_file(p, TASKS))
                except Exception:
                    continue

            best = pick_best_candidate(cands)

            row = {"iter": iter_k}
            if best is None or best["covered"] == 0:
                row["AVG"] = "-"
                for t in TASKS:
                    row[t] = "-"
                incomplete_report.append((iter_k, TASKS))
            else:
                ts = best["task_success"]
                row["AVG"] = fmt_pct(best["avg_success"])
                for t in TASKS:
                    row[t] = fmt_pct(ts.get(t))
                if not best["is_complete"]:
                    incomplete_report.append((iter_k, best["missing"]))

            table_rows.append(row)

        table_rows.sort(key=lambda r: iter_k_to_int(r["iter"]))

        title = f"{root} ({dir_flag})"
        if not table_rows:
            print(f"No records found for {title}\n")
            continue

        if print_style == "text":
            print("=" * 60)
            print(title)
            print(header_fmt.format(*headers))
            for r in table_rows:
                print(row_fmt.format(**r))
            print()
        else:
            # ================== MD pivot table ==================
            iters = [r["iter"] for r in table_rows]

            # AVG
            metrics = TASKS + ["AVG"]

            print(f"**{title}**\n")

            # header
            print("| Task | " + " | ".join(iters) + " |")
            print("| --- | " + " | ".join("---" for _ in iters) + " |")

            # rows
            for m in metrics:
                row = [m] + [r[m] for r in table_rows]
                print("| " + " | ".join(row) + " |")

            print()

        # best AVG
        valid = []
        for r in table_rows:
            if r["AVG"] != "-":
                try:
                    valid.append((float(r["AVG"]), r))
                except ValueError:
                    pass
        if valid:
            best_avg, best_row = max(valid, key=lambda x: x[0])
            print(f"--> Best AVG: iter={best_row['iter']}  AVG={best_row['AVG']}%")
        else:
            print("--> No valid AVG found.")

        if incomplete_report:
            print("--> Incomplete iters (missing tasks):")
            for it, miss in sorted(incomplete_report, key=lambda x: iter_k_to_int(x[0])):
                print(f"    - {it}: {','.join(miss)}")

        print()


if __name__ == "__main__":
    roots = [
        "/path/to/experiment1",
        "/path/to/experiment2",
    ]
    extract_and_print_mikasa(roots, dir_flag="eval_mikasa", print_style="md")
