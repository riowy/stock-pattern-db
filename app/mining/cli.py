"""CLI formatting for ephemeral mining results."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from app.mining.engine import PatternResult


def print_audit_table(console: Console, rows: list[dict], *, title: str) -> None:
    if not rows:
        return
    table = Table(title=title)
    table.add_column("subset")
    table.add_column("n")
    table.add_column("median")
    table.add_column("winsor")
    table.add_column("NW t")
    table.add_column("vs_base")
    for row in rows:
        table.add_row(
            str(row.get("subset", "")),
            str(row.get("n", "")),
            _f(row.get("excess_median") if row.get("excess_median") is not None else row.get("median")),
            _f(row.get("winsor_mean")),
            _f(row.get("newey_west_t")),
            _f(row.get("vs_baseline_mean")),
        )
    console.print(table)
    console.print("[dim]Audit only. Does not retune pattern thresholds.[/dim]")


def print_mining_table(console: Console, rows: list[PatternResult], *, title: str, top: int) -> None:
    console.print(f"[bold]{title}[/bold]")
    console.print(
        "pattern | family | a_n | a_ep | a_med_ex | a_vs_base | a_NW | q | v_n | v_med_ex | v_vs_base | same | overlap"
    )
    shown = 0
    for row in rows:
        if shown >= top:
            break
        a = row.analysis
        v = row.validation or {}
        console.print(
            " | ".join(
                [
                    row.pattern,
                    row.family,
                    str(a.get("kept", a.get("n"))),
                    str(a.get("episodes", "")),
                    _f(a.get("excess_median") if a.get("excess_median") is not None else a.get("median")),
                    _f(a.get("vs_baseline_mean")),
                    _f(a.get("vs_baseline_nw_t") if a.get("vs_baseline_nw_t") is not None else a.get("newey_west_t")),
                    _f(row.fdr_q),
                    str(v.get("kept", v.get("n", ""))),
                    _f(v.get("excess_median") if v.get("excess_median") is not None else v.get("median")),
                    _f(v.get("vs_baseline_mean")),
                    "" if row.same_direction is None else ("Y" if row.same_direction else "n"),
                    row.overlap_flag or "",
                ]
            )
        )
        shown += 1
    console.print("[dim]Research statistics only. Not buy/sell recommendations. Results are not saved.[/dim]")
    console.print("[dim]FDR uses same-date CS contrast. VALIDATION does not retune. FUTURE_HOLDOUT is not evaluated.[/dim]")


def print_inspect(console: Console, report: dict) -> None:
    console.print(f"[bold]inspect | {report['pattern']}[/bold]")
    for key, block in report.get("modes", {}).items():
        a = block["analysis"]
        v = block["validation"]
        conc = block["concentration"]
        loo = block["leave_one_security"]
        console.print(
            f"  mode={key} analysis kept={a.get('kept')} rows={a.get('rows')} entry={a.get('entry_events')} "
            f"episodes={a.get('episodes')} dates={a.get('distinct_dates')} secs={a.get('distinct_securities')} "
            f"med={_f(a.get('excess_median') or a.get('median'))} vs_base={_f(a.get('vs_baseline_mean'))} "
            f"NW={_f(a.get('vs_baseline_nw_t') or a.get('newey_west_t'))}"
        )
        console.print(
            f"           validation kept={v.get('kept')} rows={v.get('rows')} entry={v.get('entry_events')} "
            f"episodes={v.get('episodes')} med={_f(v.get('excess_median') or v.get('median'))} "
            f"vs_base={_f(v.get('vs_baseline_mean'))} NW={_f(v.get('vs_baseline_nw_t') or v.get('newey_west_t'))}"
        )
        console.print(
            f"           top1_share={_f(conc.get('top_1_security_share'))} top5_share={_f(conc.get('top_5_security_share'))} "
            f"top={conc.get('top_security')} LOO drop={loo.get('dropped')} "
            f"full_med={_f(loo.get('full_median'))} dropped_med={_f(loo.get('dropped_median'))} "
            f"delta={_f(loo.get('delta_median'))}"
        )
        years = [y for y in block.get("years", []) if y.get("n")]
        if years:
            console.print(
                "           years: "
                + " ; ".join(f"{y['year']} n={y['n']} med={_f(y['median_excess'])} pos={_f(y['positive_ratio'])}" for y in years)
            )
    inc = report.get("incremental")
    if inc:
        console.print(
            f"  pair incremental | parents {inc['parent_a']} , {inc['parent_b']}"
        )
        for label, side in (("vs_A", inc["vs_a"]), ("vs_B", inc["vs_b"])):
            console.print(
                f"    {label}: parent_n={side.get('parent_n')} parent_med={_f(side.get('parent_median'))} "
                f"inc_med={_f(side.get('incremental_median'))} inc_win={_f(side.get('incremental_winsor_mean'))} "
                f"inc_daily={_f(side.get('incremental_daily_mean'))} inc_NW={_f(side.get('incremental_nw_t'))}"
            )
    ov = report.get("overlap") or {}
    if ov:
        console.print(
            f"  overlap jaccard={_f(ov.get('jaccard'))} P(B|A)={_f(ov.get('p_b_given_a'))} "
            f"P(A|B)={_f(ov.get('p_a_given_b'))} flag={ov.get('flag') or ''}"
        )
    print_audit_table(console, report.get("price_floor") or [], title=f"Price-floor | {report['pattern']}")
    print_audit_table(console, report.get("regimes") or [], title=f"Regime | {report['pattern']}")


def print_walkforward(console: Console, report: dict) -> None:
    console.print("[bold]Walk-forward folds[/bold] (cutoffs freeze on each discover window)")
    for fold in report.get("folds", []):
        if fold.get("skipped"):
            console.print(f"  {fold['fold']}: skipped ({fold['skipped']})")
            continue
        if fold.get("full_discover"):
            extra = f"candidates={fold.get('n_candidates')} selected={fold.get('n_selected')}"
        else:
            extra = f"tracked={fold.get('n_tracked', 0)} (cutoffs freeze on discover; no FDR rediscovery)"
        console.print(
            f"  {fold['fold']}: discover {fold['discover']} eval {fold['eval']} {extra}"
        )
        if fold.get("selection"):
            from app.mining.selection_diag import format_selection_lines

            for line in format_selection_lines(fold["selection"]):
                console.print(f"    {line}", markup=False)
    console.print("[bold]Tracked pattern stability[/bold]")
    console.print(
        "pattern | folds | same_dir | pos | neg | med_eff | min | max | med_NW"
    )
    for row in report.get("stability", []):
        console.print(
            " | ".join(
                [
                    row["pattern"],
                    str(row["folds_tested"]),
                    str(row["same_direction_folds"]),
                    str(row["positive_effect_folds"]),
                    str(row["negative_effect_folds"]),
                    _f(row["median_fold_effect"]),
                    _f(row["min_fold_effect"]),
                    _f(row["max_fold_effect"]),
                    _f(row["median_fold_NW_t"]),
                ]
            )
        )
        for fr in row.get("folds", []):
            ev = fr.get("eval") or {}
            reason = fr.get("selection_reason")
            if reason:
                reason_bit = f" reason={reason}"
            elif fr.get("discover_selected"):
                reason_bit = " reason=SELECTED"
            else:
                reason_bit = ""
            console.print(
                f"    {fr['fold']} selected={fr.get('discover_selected')}{reason_bit} "
                f"n={ev.get('kept', ev.get('n'))} "
                f"med={_f(fr.get('effect'))} NW={_f(fr.get('nw_t'))}"
            )
    console.print("[dim]Not a score. FUTURE_HOLDOUT is not evaluated. Results are not saved.[/dim]")


def _f(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
