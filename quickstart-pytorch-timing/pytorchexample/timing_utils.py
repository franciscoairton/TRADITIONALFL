"""Utilities for timing logs used by the Flower example.

This version avoids concurrent CSV appends. In Flower simulations, several Ray
client processes can write at the same time. Appending all rows to a single CSV
can lose/corrupt rows. Therefore, each timing event is first written to an
individual CSV file under logs/_events/exec_XXX/. At the end, the server merges
all events into one readable CSV per execution and then creates the compiled
summary files.
"""

from __future__ import annotations

import csv
import os
import shutil
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

FIELDNAMES = [
    "timestamp",
    "execution_id",
    "server_round",
    "node_id",
    "stage",
    "elapsed_sec",
    "details",
]

TOTAL_SERVER_STAGES = {
    "seleciona_clientes",
    "envia_para_clientes",
    "agregacao_de_todos_clientes",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def get_log_dir(default: str = "logs") -> Path:
    log_dir = Path(os.environ.get("FL_TIMING_LOG_DIR", default))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def exec_log_path(execution_id: int, log_dir: str = "logs") -> Path:
    return get_log_dir(log_dir) / f"fl_timing_exec_{execution_id:03d}.csv"


def event_dir_path(execution_id: int, log_dir: str = "logs") -> Path:
    return get_log_dir(log_dir) / "_events" / f"exec_{execution_id:03d}"


def clear_execution_logs(*, execution_id: int, log_dir: str = "logs") -> None:
    """Remove previous logs for one execution.

    On Windows, Ray/Python can keep file handles open briefly. If deletion is
    blocked, the old event folder is renamed and a new clean folder is created,
    so the run does not fail with WinError 5.
    """
    csv_path = exec_log_path(execution_id, log_dir)
    if csv_path.exists():
        try:
            csv_path.unlink()
        except OSError:
            archived_csv = csv_path.with_name(f"{csv_path.stem}_old_{int(time.time())}{csv_path.suffix}")
            try:
                csv_path.rename(archived_csv)
            except OSError:
                pass

    ev_dir = event_dir_path(execution_id, log_dir)
    ev_dir.parent.mkdir(parents=True, exist_ok=True)
    if ev_dir.exists():
        removed = False
        for _ in range(5):
            try:
                shutil.rmtree(ev_dir)
                removed = True
                break
            except (PermissionError, OSError):
                time.sleep(0.5)
        if not removed:
            archived = ev_dir.with_name(f"{ev_dir.name}_old_{int(time.time())}")
            try:
                ev_dir.rename(archived)
            except OSError:
                pass

    ev_dir.mkdir(parents=True, exist_ok=True)


def _row_sort_key(row: dict):
    ts = parse_iso(row.get("timestamp", "")) or datetime.min
    server_round = safe_int(row.get("server_round", "0")) or 0
    return (server_round, ts, row.get("stage", ""), str(row.get("node_id", "")))


def write_log(
    *,
    execution_id: int,
    server_round: int,
    node_id: str | int,
    stage: str,
    elapsed_sec: float,
    details: str = "",
    log_dir: str = "logs",
) -> None:
    """Write one timing event using one file per event.

    This is safer than appending to a shared CSV because Ray client processes can
    write concurrently. The final per-execution CSV is created by
    materialize_execution_logs().
    """
    ev_dir = event_dir_path(execution_id, log_dir)
    ev_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": now_iso(),
        "execution_id": execution_id,
        "server_round": server_round,
        "node_id": node_id,
        "stage": stage,
        "elapsed_sec": f"{elapsed_sec:.9f}",
        "details": details,
    }
    filename = f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex}.csv"
    path = ev_dir / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=_row_sort_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def materialize_execution_logs(*, log_dir: str, execution_ids: Iterable[int]) -> None:
    """Merge per-event files into one readable CSV per execution."""
    for execution_id in execution_ids:
        ev_dir = event_dir_path(execution_id, log_dir)
        rows: list[dict] = []
        if ev_dir.exists():
            for path in ev_dir.glob("*.csv"):
                rows.extend(_read_rows(path))
        # Keep support for logs generated by older versions.
        old_rows = _read_rows(exec_log_path(execution_id, log_dir))
        if old_rows and not rows:
            rows.extend(old_rows)
        if rows:
            _write_rows(exec_log_path(execution_id, log_dir), rows)


def iter_log_rows(log_dir: str, execution_ids: Iterable[int]) -> Iterable[dict]:
    for execution_id in execution_ids:
        path = exec_log_path(execution_id, log_dir)
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or row.get("stage") is None:
                    continue
                row.setdefault("execution_id", str(execution_id))
                yield row


def ensure_estimated_send_rows(*, log_dir: str, execution_ids: Iterable[int]) -> None:
    """Insert envia_para_clientes rows when Flower internals did not expose them.

    The value is estimated using server timestamps:

        envia_para_clientes ~= inicio_agregacao - fim_selecao

    This stage already includes sending the model, waiting for client training,
    receiving the replies, and Flower/Ray communication overhead. Therefore, it
    should be used for total round time instead of summing all client training
    rows.
    """
    for execution_id in execution_ids:
        path = exec_log_path(execution_id, log_dir)
        rows = _read_rows(path)
        if not rows:
            continue

        changed = False
        by_round: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            server_round = safe_int(row.get("server_round", "0"))
            if server_round is None or server_round <= 0:
                continue
            by_round[server_round].append(row)

        for server_round, round_rows in by_round.items():
            stages = {row.get("stage", "") for row in round_rows}
            if "envia_para_clientes" in stages:
                continue

            selection_rows = [r for r in round_rows if r.get("stage") == "seleciona_clientes"]
            aggregation_rows = [r for r in round_rows if r.get("stage") == "agregacao_de_todos_clientes"]
            if not selection_rows or not aggregation_rows:
                continue

            selection = selection_rows[0]
            aggregation = aggregation_rows[-1]
            t_selection_end = parse_iso(selection.get("timestamp", ""))
            t_aggregation_end = parse_iso(aggregation.get("timestamp", ""))
            aggregation_elapsed = safe_float(aggregation.get("elapsed_sec", "")) or 0.0
            if t_selection_end is None or t_aggregation_end is None:
                continue

            estimated = (t_aggregation_end - t_selection_end).total_seconds() - aggregation_elapsed
            if estimated < 0:
                estimated = 0.0

            rows.append(
                {
                    "timestamp": aggregation.get("timestamp", now_iso()),
                    "execution_id": str(execution_id),
                    "server_round": str(server_round),
                    "node_id": "server",
                    "stage": "envia_para_clientes",
                    "elapsed_sec": f"{estimated:.9f}",
                    "details": (
                        "estimado_por_timestamps; inclui envio, treino nos clientes, retorno e overhead Flower/Ray; "
                        "calculado como inicio_agregacao - fim_selecao"
                    ),
                }
            )
            changed = True

        if changed:
            _write_rows(path, rows)


def _t_critical_95_two_tailed(df: int) -> float:
    """Return t critical value for a 95% two-tailed confidence interval.

    A small built-in table avoids adding scipy as a dependency. For df > 30,
    the normal approximation 1.96 is used.
    """
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(df, 1.96)


def mean_std_ci95(values: list[float]) -> tuple[float, float, float, float]:
    """Return mean, sample std dev, ci95 low, and ci95 high."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, mean, mean
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std_dev = variance ** 0.5
    margin = _t_critical_95_two_tailed(n - 1) * (std_dev / (n ** 0.5))
    return mean, std_dev, mean - margin, mean + margin


def compile_summary(
    *,
    log_dir: str,
    execution_ids: list[int],
    warmup_rounds: int,
) -> None:
    """Generate only the compiled summary with warmup included.

    The project no longer generates compiled_without_warmup.csv. The output file
    compiled_with_warmup.csv includes the mean total time per execution and a
    95% confidence interval computed across the N executions.
    """
    materialize_execution_logs(log_dir=log_dir, execution_ids=execution_ids)
    ensure_estimated_send_rows(log_dir=log_dir, execution_ids=execution_ids)
    rows = list(iter_log_rows(log_dir, execution_ids))
    if not rows:
        return

    # Remove the old file, if it exists from a previous version/run.
    old_without = get_log_dir(log_dir) / "compiled_without_warmup.csv"
    if old_without.exists():
        try:
            old_without.unlink()
        except OSError:
            pass

    _write_summary_file(
        rows=rows,
        log_dir=log_dir,
        output_name="compiled_with_warmup.csv",
        execution_ids=execution_ids,
        warmup_rounds=warmup_rounds,
    )


def _write_summary_file(
    *,
    rows: list[dict],
    log_dir: str,
    output_name: str,
    execution_ids: list[int],
    warmup_rounds: int,
) -> None:
    """Write a summary CSV with mean times by stage and total-time CI."""
    stage_values: dict[str, list[float]] = defaultdict(list)
    round_totals: dict[tuple[int, int], float] = defaultdict(float)
    wallclock_total_by_execution: dict[int, float] = {}

    for row in rows:
        elapsed = safe_float(row.get("elapsed_sec", ""))
        if elapsed is None:
            continue

        stage = row.get("stage", "")
        execution_id = safe_int(row.get("execution_id", "0"))
        server_round = safe_int(row.get("server_round", "0"))
        if execution_id is None or server_round is None:
            continue

        if stage == "tempo_total_execucao":
            wallclock_total_by_execution[execution_id] = elapsed
            stage_values[stage].append(elapsed)
            continue

        stage_values[stage].append(elapsed)

        if server_round > 0 and stage in TOTAL_SERVER_STAGES:
            round_totals[(execution_id, server_round)] += elapsed

    round_total_values = [v for v in round_totals.values() if v > 0]
    round_mean = sum(round_total_values) / len(round_total_values) if round_total_values else 0.0

    total_by_execution: dict[int, float] = defaultdict(float)
    rounds_by_execution: dict[int, int] = defaultdict(int)
    for (execution_id, _server_round), value in round_totals.items():
        total_by_execution[execution_id] += value
        rounds_by_execution[execution_id] += 1

    execution_total_values = [
        total_by_execution[eid]
        for eid in execution_ids
        if total_by_execution.get(eid, 0) > 0
    ]
    execution_total_mean, execution_total_std, execution_total_ci_low, execution_total_ci_high = mean_std_ci95(
        execution_total_values
    )

    wallclock_values = [wallclock_total_by_execution[eid] for eid in execution_ids if eid in wallclock_total_by_execution]
    wallclock_mean, wallclock_std, wallclock_ci_low, wallclock_ci_high = mean_std_ci95(wallclock_values)

    out_path = get_log_dir(log_dir) / output_name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "summary_type",
            "stage",
            "mean_elapsed_sec",
            "count",
            "std_dev_sec",
            "ci95_low_sec",
            "ci95_high_sec",
            "warmup_rounds_removed",
            "details",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow(
            {
                "summary_type": "total",
                "stage": "tempo_total_medio_por_execucao_com_warmup",
                "mean_elapsed_sec": f"{execution_total_mean:.9f}",
                "count": len(execution_total_values),
                "std_dev_sec": f"{execution_total_std:.9f}",
                "ci95_low_sec": f"{execution_total_ci_low:.9f}",
                "ci95_high_sec": f"{execution_total_ci_high:.9f}",
                "warmup_rounds_removed": 0,
                "details": (
                    "media entre as N execucoes da soma das rodadas usando somente etapas server-side: "
                    "seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes; "
                    "IC 95% calculado sobre os totais das N execucoes"
                ),
            }
        )
        writer.writerow(
            {
                "summary_type": "total",
                "stage": "tempo_total_medio_por_rodada_com_warmup",
                "mean_elapsed_sec": f"{round_mean:.9f}",
                "count": len(round_total_values),
                "std_dev_sec": "",
                "ci95_low_sec": "",
                "ci95_high_sec": "",
                "warmup_rounds_removed": 0,
                "details": (
                    "media das rodadas usando seleciona_clientes + envia_para_clientes + agregacao_de_todos_clientes; "
                    "nao soma tempos dos clientes separadamente porque eles executam em paralelo e ja estao embutidos em envia_para_clientes"
                ),
            }
        )
        if wallclock_values:
            writer.writerow(
                {
                    "summary_type": "total",
                    "stage": "tempo_wallclock_medio_strategy_start_com_warmup",
                    "mean_elapsed_sec": f"{wallclock_mean:.9f}",
                    "count": len(wallclock_values),
                    "std_dev_sec": f"{wallclock_std:.9f}",
                    "ci95_low_sec": f"{wallclock_ci_low:.9f}",
                    "ci95_high_sec": f"{wallclock_ci_high:.9f}",
                    "warmup_rounds_removed": 0,
                    "details": "tempo wall-clock medido ao redor de strategy.start; inclui avaliacoes globais e overheads do Flower",
                }
            )

        for stage in sorted(stage_values):
            values = stage_values[stage]
            mean = sum(values) / len(values)
            writer.writerow(
                {
                    "summary_type": "stage",
                    "stage": stage,
                    "mean_elapsed_sec": f"{mean:.9f}",
                    "count": len(values),
                    "std_dev_sec": "",
                    "ci95_low_sec": "",
                    "ci95_high_sec": "",
                    "warmup_rounds_removed": 0,
                    "details": "media aritmetica dos registros da etapa; warmup incluido",
                }
            )
