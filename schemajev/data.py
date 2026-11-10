"""Validate the paper's data scope and split the canonical training index."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import tempfile

DATA = Path(__file__).resolve().parents[1] / "data"
HEX = re.compile(r"[0-9a-f]{64}")
FIELDS = ("dataset_id", "row_index", "native_record_id", "task_type", "canonical_input_sha256")


def partition(input_digest):
    if not isinstance(input_digest, str) or not HEX.fullmatch(input_digest):
        raise ValueError("expected the canonical input SHA-256")
    salt = f"514|H2-global-exact-input-FIT80-DEV20-v1|{input_digest}"
    return "DEV" if int(hashlib.sha256(salt.encode()).hexdigest(), 16) < (1 << 256) // 5 else "FIT"


def split(index, output):
    with (DATA / "train_views.csv").open(newline="") as f:
        scope = {r["dataset_id"]: r for r in csv.DictReader(f)}
    counts = Counter()
    seen = set()
    if output.exists():
        raise ValueError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        payload = Path(staging) / "split"
        payload.mkdir()
        with index.open() as source, (payload / "FIT.jsonl").open("w") as fit, (payload / "DEV.jsonl").open("w") as dev:
            position = 0
            for line in source:
                obj = json.loads(line)
                for row in obj.get("entries", [obj]):
                    item = {k: row[k] for k in FIELDS}
                    key, idx = item["dataset_id"], item["row_index"]
                    meta = scope[key]
                    if (type(idx) is not int or not 0 <= idx < int(meta["fit_rows"]) + int(meta["dev_rows"])
                        or (key, idx) in seen or item["task_type"] != meta["task_type"]
                        or not isinstance(item["native_record_id"], str) or not item["native_record_id"]):
                        raise ValueError(f"invalid training occurrence: {key}:{idx}")
                    seen.add((key, idx))
                    role = partition(item["canonical_input_sha256"])
                    counts[key, role] += 1
                    item["original_course_position"] = position
                    position += 1
                    (fit if role == "FIT" else dev).write(json.dumps(item, ensure_ascii=False) + "\n")
        for key, meta in scope.items():
            for role in ("FIT", "DEV"):
                if counts[key, role] != int(meta[role.lower() + "_rows"]):
                    raise ValueError(f"paper count mismatch for {key}:{role}")
        payload.rename(output)
    print(json.dumps({r: sum(n for (_, role), n in counts.items() if role == r) for r in ("FIT", "DEV")}))


def check_panel(index):
    panel = json.loads((DATA / "eval_views.json").read_text())
    digest = hashlib.sha256()
    counts = Counter()
    views = []
    fields = ("dataset_id", "row_index", "canonical_record_id", "canonical_input_sha256", "input_ids_sha256", "option_mapping")
    with index.open() as f:
        for line in f:
            row = json.loads(line)
            identity = {k: row[k] for k in fields}
            key = identity["dataset_id"]
            if not views or views[-1] != key:
                views.append(key)
            counts[key] += 1
            digest.update((json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    if views != [r["view"] for r in panel] or counts != Counter({r["view"]: r["rows"] for r in panel}):
        raise ValueError("panel order or counts differ")
    if digest.hexdigest() != "8124ed954d29f63d03b5cc280e221b5db126abbd64ddf3fe1afc34aeb4b9f6c0":
        raise ValueError("canonical panel identities differ")
    print(json.dumps({"views": len(views), "rows": sum(counts.values()), "ordered_identity_sha256": digest.hexdigest()}))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("split", "check-panel"):
        command = commands.add_parser(name)
        command.add_argument("index", type=Path)
        if name == "split":
            command.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "split":
        split(args.index, args.output)
    else:
        check_panel(args.index)


if __name__ == "__main__":
    main()
