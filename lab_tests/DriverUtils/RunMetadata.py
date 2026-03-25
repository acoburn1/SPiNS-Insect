from datetime import datetime, timezone
from pathlib import Path
import subprocess
import json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_branch() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _compact_utc(iso_utc: str) -> str:
    return (
        iso_utc.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )


def write_stage_metadata(
    *,
    result_dir: str,
    stage_name: str,
    stage_dir: str,
    started_at_utc: str,
    finished_at_utc: str,
    status: str,
    config_path: str | None,
    details: dict | None = None,
) -> dict:
    branch = git_branch()
    commit = git_commit()
    record_id = f"{stage_name}_{_compact_utc(started_at_utc)}"

    metadata = {
        "schema_version": 1,
        "record_id": record_id,
        "stage": stage_name,
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "code": {
            "git_branch": branch,
            "git_commit": commit,
        },
        "config_path": config_path,
        "stage_dir": stage_dir,
        "details": details or {},
    }

    stage_history_dir = Path(result_dir) / "RunMetadata" / stage_name
    stage_history_dir.mkdir(parents=True, exist_ok=True)
    with open(stage_history_dir / f"{record_id}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(stage_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
