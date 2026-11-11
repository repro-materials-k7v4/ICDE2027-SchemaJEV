"""Load and verify the final descriptions for the reported 18 evaluation views."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"


def canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_evaluation_views(catalog_path=None, provenance_path=None):
    """Validate the final catalog without loading raw records or model weights."""
    catalog_path = Path(catalog_path) if catalog_path is not None else DATA / "eval_views.json"
    provenance_path = (
        Path(provenance_path)
        if provenance_path is not None
        else DATA / "eval_conditions_provenance.json"
    )
    panel = json.loads(catalog_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(panel, list) or len(panel) != provenance["views"]:
        raise ValueError("evaluation view count differs from the final protocol")
    conditions = {}
    counts = Counter()
    records = 0
    for view in panel:
        key = view["view"]
        if not isinstance(key, str) or not key or key in conditions:
            raise ValueError("evaluation view IDs must be distinct nonempty strings")
        condition = view["description"]
        if (
            not isinstance(condition, dict)
            or set(condition) != {"context", "instruction"}
            or not all(isinstance(value, str) for value in condition.values())
        ):
            raise ValueError("a model condition contains only context and instruction strings")
        if canonical_sha256(condition) != view["description_sha256"]:
            raise ValueError(f"description fingerprint differs for {key}")
        if type(view["rows"]) is not int or view["rows"] <= 0:
            raise ValueError(f"invalid record count for {key}")
        conditions[key] = condition
        counts[view["cohort"], view["task_type"]] += 1
        records += view["rows"]
    expected = Counter(
        {(cohort, task_type): n for cohort, n in (("ID12", 4), ("OOD6", 2))
         for task_type in ("choice", "binary", "score")}
    )
    if counts != expected or records != provenance["records"]:
        raise ValueError("evaluation cohort, task types, or record counts differ")
    view_contract = [
        {key: view[key] for key in ("view", "cohort", "task_type", "rows", "mask_sha256")}
        for view in panel
    ]
    if canonical_sha256(view_contract) != provenance["ordered_view_contract_sha256"]:
        raise ValueError("evaluation view order or mask fingerprints differ")
    if canonical_sha256(conditions) != provenance["description_map_sha256"]:
        raise ValueError("description map differs from the final evaluation inputs")
    return panel


def load_evaluation_conditions(catalog_path=None, provenance_path=None):
    """Return view-indexed conditions; pass each value, not its key, to the model."""
    return {
        view["view"]: view["description"]
        for view in load_evaluation_views(catalog_path, provenance_path)
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--catalog", type=Path, default=DATA / "eval_views.json")
    cli.add_argument(
        "--provenance", type=Path, default=DATA / "eval_conditions_provenance.json"
    )
    args = cli.parse_args()
    panel = load_evaluation_views(args.catalog, args.provenance)
    print(json.dumps({
        "views": len(panel),
        "records": sum(view["rows"] for view in panel),
        "description_map_sha256": canonical_sha256({
            view["view"]: view["description"] for view in panel
        }),
        "verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()
