from pathlib import Path
from datetime import datetime, timezone


_WARNINGS_LOG = Path(__file__).resolve().parents[1] / "Output" / "stats_warnings.log"


def append_stats_warning(event: str, source: str, **fields) -> None:
    _WARNINGS_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    extras = "\t".join(f"{k}={v}" for k, v in fields.items())
    line = f"{ts}\tevent={event}\tsource={source or 'unknown'}"
    if extras:
        line = f"{line}\t{extras}"
    with _WARNINGS_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
