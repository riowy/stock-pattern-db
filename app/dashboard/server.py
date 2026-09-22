"""Local research dashboard (stdlib HTTP). Bind 127.0.0.1 by default."""

from __future__ import annotations

import html
import json
import urllib.parse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from app.aggregation.engine import SimpleCountAggregationEngine
from app.dashboard.source_status import (
    SourceFreshnessSummary,
    SourceStatusProvider,
    StaticSourceStatusProvider,
    unavailable_source_status,
)
from app.discovery.generators import GeneratorRegistry
from app.discovery.metrics import all_pairwise_overlaps, compute_generator_metrics, proposer_map
from app.patterns.store import PatternResearchStore
from app.signals.models import DailyPatternSignal, SignalService


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _layout(title: str, body: str, *, flash: str | None = None) -> str:
    nav = """
    <nav>
      <a href="/">Overview</a>
      <a href="/generators">Generators / AI</a>
      <a href="/patterns">Pattern Registry</a>
      <a href="/rejected">Rejected Archive</a>
      <a href="/signals">Today's Signals</a>
      <a href="/candidates">Candidate List</a>
    </nav>
    """
    flash_html = f'<div class="flash">{_esc(flash)}</div>' if flash else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_esc(title)} · Pattern Research</title>
<style>
  :root {{ --bg:#f6f4ef; --ink:#1a1a1a; --muted:#5c5c5c; --line:#d9d3c7; --accent:#245b4e; }}
  body {{ font-family: "Segoe UI", "Helvetica Neue", sans-serif; margin:0; background:var(--bg); color:var(--ink); }}
  header {{ padding:1.25rem 1.5rem; border-bottom:1px solid var(--line); background:linear-gradient(120deg,#eef2ef,#f6f4ef 55%,#efe8dc); }}
  header h1 {{ margin:0; font-size:1.35rem; letter-spacing:0.02em; }}
  header p {{ margin:0.35rem 0 0; color:var(--muted); font-size:0.95rem; }}
  nav {{ display:flex; flex-wrap:wrap; gap:0.75rem; padding:0.75rem 1.5rem; border-bottom:1px solid var(--line); background:#fff; }}
  nav a {{ color:var(--accent); text-decoration:none; font-weight:600; }}
  main {{ padding:1.25rem 1.5rem 3rem; max-width:1100px; }}
  .empty {{ padding:2rem; border:1px dashed var(--line); background:#fff; color:var(--muted); }}
  table {{ width:100%; border-collapse:collapse; background:#fff; }}
  th, td {{ text-align:left; padding:0.55rem 0.65rem; border-bottom:1px solid var(--line); vertical-align:top; font-size:0.92rem; }}
  th {{ font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em; color:var(--muted); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:0.75rem; margin:1rem 0; }}
  .card {{ background:#fff; border:1px solid var(--line); padding:0.85rem; }}
  .card .n {{ font-size:1.4rem; font-weight:700; }}
  .card .l {{ color:var(--muted); font-size:0.8rem; }}
  .flash {{ background:#e7f2ee; border:1px solid #b7d7cb; padding:0.75rem 1rem; margin-bottom:1rem; }}
  form.inline {{ display:inline; }}
  button {{ background:var(--accent); color:#fff; border:0; padding:0.35rem 0.7rem; cursor:pointer; }}
  button.secondary {{ background:#6b6358; }}
  code {{ font-size:0.85em; }}
  .muted {{ color:var(--muted); }}
  pre {{ background:#fff; border:1px solid var(--line); padding:0.75rem; overflow:auto; }}
</style>
</head>
<body>
<header>
  <h1>Pattern Research Dashboard</h1>
  <p>Local-only research inspection. Not trading advice. No broker actions.</p>
</header>
{nav}
<main>
{flash_html}
{body}
</main>
</body>
</html>"""


class DashboardApp:
    def __init__(
        self,
        store: PatternResearchStore,
        *,
        persistence_enabled: bool = False,
        signal_persistence_enabled: bool = False,
        host: str = "127.0.0.1",
        port: int = 8765,
        source_status: SourceStatusProvider | None = None,
    ) -> None:
        self.store = store
        self.persistence_enabled = persistence_enabled
        self.signal_persistence_enabled = signal_persistence_enabled
        self.host = host
        self.port = port
        self.source_status = source_status or StaticSourceStatusProvider(unavailable_source_status())
        self.generators = GeneratorRegistry(store)
        self.signals = SignalService(store)
        self.aggregator = SimpleCountAggregationEngine()

    def _render_source_freshness(self) -> str:
        summary: SourceFreshnessSummary = self.source_status.get_summary()
        rows = [
            ("Availability", summary.availability),
            ("Message", summary.message),
            ("Prices latest", summary.prices_latest or "—"),
            ("Features latest", summary.features_latest or "—"),
            ("Labels latest", summary.labels_latest or "—"),
            ("Expected latest session", summary.expected_latest_session or "—"),
        ]
        if summary.tracked_note:
            rows.append(("Note", summary.tracked_note))
        rows_html = "".join(
            f"<tr><th>{_esc(label)}</th><td>{_esc(value)}</td></tr>" for label, value in rows
        )
        return f"""
        <section class="source-freshness" aria-label="Source data freshness">
          <h2>Source data freshness</h2>
          <p class="muted">Read-only summary. No network calls and no catalog/lake writes from this page.</p>
          <table>{rows_html}</table>
        </section>
        """

    def make_handler(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
                return

            def _send(self, code: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
                data = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)
                flash = qs.get("flash", [None])[0]
                try:
                    if path == "/":
                        self._send(200, app.render_overview(flash=flash))
                    elif path == "/generators":
                        self._send(200, app.render_generators(flash=flash))
                    elif path.startswith("/generators/"):
                        gid = urllib.parse.unquote(path.split("/", 2)[2])
                        self._send(200, app.render_generator_detail(gid, flash=flash))
                    elif path == "/patterns":
                        self._send(200, app.render_patterns(qs, flash=flash))
                    elif path.startswith("/patterns/"):
                        pid = urllib.parse.unquote(path.split("/", 2)[2])
                        self._send(200, app.render_pattern_detail(pid, flash=flash))
                    elif path == "/rejected":
                        self._send(200, app.render_rejected(flash=flash))
                    elif path == "/signals":
                        self._send(200, app.render_signals(qs, flash=flash))
                    elif path == "/candidates":
                        self._send(200, app.render_candidates(qs, flash=flash))
                    elif path == "/health":
                        self._send(200, json.dumps({"ok": True}), "application/json")
                    else:
                        self._send(404, _layout("Not found", "<p>Not found.</p>"))
                except Exception as exc:  # noqa: BLE001
                    self._send(500, _layout("Error", f"<pre>{_esc(exc)}</pre>"))

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                form = parse_qs(raw)
                confirm = form.get("confirm", [""])[0] == "yes"
                action = form.get("action", [""])[0]
                generator_id = form.get("generator_id", [""])[0]

                if path == "/generators/action":
                    if not confirm:
                        self.send_response(303)
                        self.send_header(
                            "Location",
                            "/generators?flash=" + urllib.parse.quote("Action cancelled: confirmation required."),
                        )
                        self.end_headers()
                        return
                    try:
                        if action == "disable":
                            app.generators.disable(generator_id, actor="dashboard")
                            msg = f"Disabled generator {generator_id} (history preserved)."
                        elif action == "retire":
                            app.generators.retire(generator_id, actor="dashboard")
                            msg = (
                                f"Retired generator {generator_id}. "
                                "This is a soft retire — history is preserved, not hard-deleted."
                            )
                        elif action == "restore":
                            app.generators.restore(generator_id, actor="dashboard")
                            msg = f"Restored generator {generator_id} to ACTIVE."
                        else:
                            msg = "Unknown action."
                    except Exception as exc:  # noqa: BLE001
                        msg = f"Action failed: {exc}"
                    self.send_response(303)
                    self.send_header("Location", "/generators?flash=" + urllib.parse.quote(msg))
                    self.end_headers()
                    return

                self._send(404, _layout("Not found", "<p>Not found.</p>"))

        return Handler

    def serve_forever(self) -> None:
        handler = self.make_handler()
        server = ThreadingHTTPServer((self.host, self.port), handler)
        print(f"Pattern research dashboard at http://{self.host}:{self.port}/ (Ctrl+C to stop)")
        server.serve_forever()

    # --- pages --------------------------------------------------------------------

    def render_overview(self, *, flash: str | None = None) -> str:
        freshness_html = self._render_source_freshness()
        if not self.persistence_enabled and not self.store.list_generators() and not self.store.list_patterns():
            body = freshness_html + """
            <div class="empty">
              <p><strong>Empty state.</strong> Pattern registry persistence is disabled
              (<code>PATTERN_REGISTRY_PERSISTENCE_ENABLED=false</code>) and no in-memory fixture data is loaded.</p>
              <p>Enable persistence explicitly, or launch the dashboard with a temporary fixture DB for local inspection.</p>
            </div>
            """
            return _layout("Overview", body, flash=flash)

        gens = self.store.list_generators()
        by_status: dict[str, int] = {"ACTIVE": 0, "DISABLED": 0, "RETIRED": 0}
        for g in gens:
            by_status[g["status"]] = by_status.get(g["status"], 0) + 1
        pcounts = self.store.status_counts()
        today = date.today()
        sig_n = self.store.signal_counts_for_date(today)
        # Also try fixture date if today empty
        all_sigs = self.store.list_signals()
        if sig_n == 0 and all_sigs:
            today = all_sigs[0]["signal_date"]
            sig_n = self.store.signal_counts_for_date(today)

        from app.signals.models import DailyPatternSignal

        signals_raw = self.store.list_signals(today)
        signals = [
            DailyPatternSignal(
                signal_date=s["signal_date"],
                security_id=s["security_id"],
                ticker=s.get("ticker"),
                pattern_id=s["pattern_id"],
                pattern_version=s["pattern_version"],
                direction=s["direction"],
                expected_horizon=s.get("expected_horizon"),
                event_mode=s.get("event_mode"),
                historical_sample_size=s.get("historical_sample_size"),
                hit_rate=s.get("hit_rate"),
                hit_rate_success_rule=s.get("hit_rate_success_rule"),
                median_outcome=s.get("median_outcome"),
                mean_outcome=s.get("mean_outcome"),
                typical_loss_when_wrong=s.get("typical_loss_when_wrong"),
            )
            for s in signals_raw
        ]
        cands = self.aggregator.aggregate(signals)
        label_counts: dict[str, int] = {}
        for c in cands:
            label_counts[str(c.label)] = label_counts.get(str(c.label), 0) + 1

        cards = [
            ("Generators ACTIVE", by_status.get("ACTIVE", 0)),
            ("DISABLED", by_status.get("DISABLED", 0)),
            ("RETIRED", by_status.get("RETIRED", 0)),
            ("Total patterns", sum(pcounts.values())),
        ]
        for st in ("DISCOVERED", "VALIDATING", "PASSED", "REJECTED", "MONITORING", "DEGRADED", "RETIRED"):
            cards.append((st, pcounts.get(st, 0)))
        cards.append((f"Signals ({today})", sig_n))
        for lab in ("BUY_CANDIDATE", "SELL_CANDIDATE", "CONFLICTING", "WATCH"):
            cards.append((lab, label_counts.get(lab, 0)))

        cards_html = '<div class="cards">' + "".join(
            f'<div class="card"><div class="n">{_esc(n)}</div><div class="l">{_esc(label)}</div></div>'
            for label, n in cards
        ) + "</div>"
        persist_note = (
            f"<p class='muted'>Registry persistence: "
            f"{'ON' if self.persistence_enabled else 'OFF'} · "
            f"Signal persistence: {'ON' if self.signal_persistence_enabled else 'OFF'}</p>"
        )
        return _layout("Overview", freshness_html + persist_note + cards_html, flash=flash)

    def render_generators(self, *, flash: str | None = None) -> str:
        gens = self.store.list_generators()
        if not gens:
            return _layout(
                "Generators",
                '<div class="empty"><p>No generators registered.</p></div>',
                flash=flash,
            )
        overlaps = all_pairwise_overlaps(self.store)
        rows = []
        for g in gens:
            m = compute_generator_metrics(self.store, g["generator_id"])
            rows.append(
                f"<tr>"
                f"<td><a href='/generators/{_esc(g['generator_id'])}'>{_esc(g['name'])}</a><br/>"
                f"<code>{_esc(g['generator_id'])}</code></td>"
                f"<td>{_esc(g['generator_type'])}<br/>{_esc(g['status'])}</td>"
                f"<td>{m.total_proposals} / {m.unique_patterns_proposed}<br/>"
                f"new={m.new_pattern_count} dup={m.duplicate_count}</td>"
                f"<td>rej-hit={m.known_rejected_rediscovery_count}<br/>"
                f"pass-hit={m.known_passed_rediscovery_count}</td>"
                f"<td>pass={m.passed_count} ({m.passed_rate:.0%}) "
                f"rej={m.rejected_count} ({m.rejected_rate:.0%})</td>"
                f"<td>runtime={m.total_runtime_seconds:.1f}s<br/>"
                f"avoided={m.validations_avoided}</td>"
                f"<td>{self._action_forms(g['generator_id'], g['status'])}</td>"
                f"</tr>"
            )
        overlap_rows = "".join(
            f"<tr><td>{_esc(o.generator_a)}</td><td>{_esc(o.generator_b)}</td>"
            f"<td>{o.shared_pattern_count}</td><td>{o.jaccard_proposed:.3f}</td>"
            f"<td>{o.overlap_rejected}</td><td>{o.overlap_passed}</td>"
            f"<td>{o.unique_accepted_a}/{o.unique_accepted_b}</td>"
            f"<td>{o.redundant_discovery_rate:.3f}</td></tr>"
            for o in overlaps
        )
        body = f"""
        <p class="muted">Manual Disable / Retire / Restore only. No automatic deletion.
        Retire is soft — history remains queryable.</p>
        <table>
          <tr><th>Generator</th><th>Type / Status</th><th>Proposals / Unique</th>
          <th>Rediscoveries</th><th>Outcomes</th><th>Compute</th><th>Actions</th></tr>
          {''.join(rows)}
        </table>
        <h2>Overlap / synergy</h2>
        <table>
          <tr><th>A</th><th>B</th><th>Shared</th><th>Jaccard</th><th>Rej∩</th><th>Pass∩</th>
          <th>Unique accepted A/B</th><th>Redundant rate</th></tr>
          {overlap_rows or '<tr><td colspan="8" class="muted">Need ≥2 generators.</td></tr>'}
        </table>
        """
        return _layout("Generators", body, flash=flash)

    def _action_forms(self, generator_id: str, status: str) -> str:
        gid = _esc(generator_id)
        forms = []
        if status == "ACTIVE":
            forms.append(
                f"<form class='inline' method='post' action='/generators/action'>"
                f"<input type='hidden' name='generator_id' value='{gid}'/>"
                f"<input type='hidden' name='action' value='disable'/>"
                f"<input type='hidden' name='confirm' value='yes'/>"
                f"<button type='submit' onclick=\"return confirm('Disable {gid}? History is preserved.')\">Disable</button>"
                f"</form> "
                f"<form class='inline' method='post' action='/generators/action'>"
                f"<input type='hidden' name='generator_id' value='{gid}'/>"
                f"<input type='hidden' name='action' value='retire'/>"
                f"<input type='hidden' name='confirm' value='yes'/>"
                f"<button class='secondary' type='submit' "
                f"onclick=\"return confirm('Retire {gid}? Soft retire only — history preserved, not deleted.')\">"
                f"Retire</button></form>"
            )
        else:
            forms.append(
                f"<form class='inline' method='post' action='/generators/action'>"
                f"<input type='hidden' name='generator_id' value='{gid}'/>"
                f"<input type='hidden' name='action' value='restore'/>"
                f"<input type='hidden' name='confirm' value='yes'/>"
                f"<button type='submit' onclick=\"return confirm('Restore {gid} to ACTIVE?')\">Restore</button>"
                f"</form>"
            )
        return " ".join(forms)

    def render_generator_detail(self, generator_id: str, *, flash: str | None = None) -> str:
        g = self.store.get_generator(generator_id)
        if g is None:
            return _layout("Generator", f"<p>Unknown generator {_esc(generator_id)}</p>", flash=flash)
        runs = self.store.list_discovery_runs(generator_id)
        events = self.store.list_discovery_events(generator_id=generator_id)
        m = compute_generator_metrics(self.store, generator_id)
        versions = self.store.list_generator_versions(generator_id)
        run_html = "".join(
            f"<li><code>{_esc(r['run_id'])}</code> · {_esc(r['started_at'])} · "
            f"proposed={_esc(r.get('proposed_pattern_count'))}</li>"
            for r in runs
        )
        evt_html = "".join(
            f"<tr><td>{_esc(e['timestamp'])}</td><td>{_esc(e['proposal_kind'])}</td>"
            f"<td><a href='/patterns/{_esc(e.get('pattern_id'))}'>{_esc(e.get('pattern_id'))}</a></td>"
            f"<td>{'yes' if e.get('validation_skipped') else 'no'}</td></tr>"
            for e in events
        )
        body = f"""
        <h2>{_esc(g['name'])}</h2>
        <p>{_esc(g['generator_type'])} · {_esc(g['status'])} · <code>{_esc(generator_id)}</code></p>
        <pre>{_esc(json.dumps(m.as_dict(), indent=2, default=str))}</pre>
        <h3>Versions (immutable)</h3>
        <pre>{_esc(json.dumps(versions, indent=2, default=str))}</pre>
        <h3>DiscoveryRuns</h3>
        <ul>{run_html or '<li class="muted">None</li>'}</ul>
        <h3>DiscoveryEvents → Patterns</h3>
        <table><tr><th>When</th><th>Kind</th><th>Pattern</th><th>Validation skipped</th></tr>
        {evt_html}</table>
        """
        return _layout("Generator detail", body, flash=flash)

    def render_patterns(self, qs: dict, *, flash: str | None = None) -> str:
        status = (qs.get("status") or [None])[0]
        search = (qs.get("q") or [None])[0]
        patterns = self.store.list_patterns(status=status, search=search)
        proposers = proposer_map(self.store)
        if not patterns:
            return _layout(
                "Patterns",
                '<div class="empty"><p>No patterns in registry.</p></div>',
                flash=flash,
            )
        rows = "".join(
            f"<tr><td><a href='/patterns/{_esc(p['pattern_id'])}'>{_esc(p.get('name') or p['pattern_id'])}</a></td>"
            f"<td>{_esc(p['status'])}</td><td>{_esc(p['direction'])}</td>"
            f"<td>{_esc(p['target'])} / {_esc(p['horizon'])}</td>"
            f"<td><code>{_esc(p['hypothesis_fingerprint'][:12])}…</code></td>"
            f"<td>{_esc(', '.join(proposers.get(p['pattern_id'], [])))}</td>"
            f"<td>{_esc(p['rediscovery_count'])}</td></tr>"
            for p in patterns
        )
        body = f"""
        <form method="get" action="/patterns">
          <input name="q" placeholder="search" value="{_esc(search or '')}"/>
          <input name="status" placeholder="status filter" value="{_esc(status or '')}"/>
          <button type="submit">Filter</button>
        </form>
        <table>
          <tr><th>Pattern</th><th>Status</th><th>Dir</th><th>Target/Horizon</th>
          <th>Hyp FP</th><th>Discoverers</th><th>Rediscoveries</th></tr>
          {rows}
        </table>
        """
        return _layout("Pattern Registry", body, flash=flash)

    def render_pattern_detail(self, pattern_id: str, *, flash: str | None = None) -> str:
        rec = self.store.get_pattern_version(pattern_id)
        if rec is None:
            return _layout("Pattern", f"<p>Unknown pattern {_esc(pattern_id)}</p>", flash=flash)
        history = self.store.list_status_history(pattern_id)
        events = self.store.list_discovery_events(pattern_id=pattern_id)
        evals = self.store.list_evaluations(pattern_id)
        metrics_blocks = []
        for ev in evals:
            mets = self.store.list_metrics(ev["evaluation_id"])
            metrics_blocks.append(
                f"<h4>{_esc(ev['evaluation_id'])} · {_esc(ev['split_role'])} "
                f"{_esc(ev['period_start'])}→{_esc(ev['period_end'])} · {_esc(ev.get('decision'))}</h4>"
                f"<pre>{_esc(json.dumps(mets, indent=2, default=str))}</pre>"
            )
        body = f"""
        <h2>{_esc(rec.definition.name or pattern_id)}</h2>
        <p>Status {_esc(rec.status)} · v{_esc(rec.version)} · rediscoveries={_esc(rec.rediscovery_count)}</p>
        <p>structural <code>{_esc(rec.structural_fingerprint)}</code><br/>
           hypothesis <code>{_esc(rec.hypothesis_fingerprint)}</code></p>
        <h3>Canonical definition</h3>
        <pre>{_esc(rec.definition.model_dump_json(indent=2))}</pre>
        <h3>Status history</h3>
        <pre>{_esc(json.dumps(history, indent=2, default=str))}</pre>
        <h3>Discovery provenance</h3>
        <pre>{_esc(json.dumps(events, indent=2, default=str))}</pre>
        <h3>Evaluations / metrics</h3>
        {''.join(metrics_blocks) or '<p class="muted">None</p>'}
        """
        return _layout("Pattern detail", body, flash=flash)

    def render_rejected(self, *, flash: str | None = None) -> str:
        patterns = self.store.list_patterns(status="REJECTED")
        proposers = proposer_map(self.store)
        if not patterns:
            return _layout(
                "Rejected",
                '<div class="empty"><p>No rejected patterns (archive empty).</p></div>',
                flash=flash,
            )
        rows = "".join(
            f"<tr><td><a href='/patterns/{_esc(p['pattern_id'])}'>{_esc(p.get('name') or p['pattern_id'])}</a></td>"
            f"<td>{_esc(p.get('rejection_reason'))}</td>"
            f"<td>{_esc(p['rediscovery_count'])}</td>"
            f"<td>{_esc(', '.join(proposers.get(p['pattern_id'], [])))}</td></tr>"
            for p in patterns
        )
        body = f"""
        <p class="muted">Rejected patterns are archived forever — never hidden or hard-deleted.</p>
        <table>
          <tr><th>Pattern</th><th>Rejection reason</th><th>Rediscoveries</th><th>Generators still hitting</th></tr>
          {rows}
        </table>
        """
        return _layout("Rejected Archive", body, flash=flash)

    def render_signals(self, qs: dict, *, flash: str | None = None) -> str:
        all_sigs = self.store.list_signals()
        if not all_sigs:
            return _layout(
                "Signals",
                '<div class="empty"><p>No signals. Production daily signal generation is not enabled in v1.</p></div>',
                flash=flash,
            )
        d = (qs.get("date") or [None])[0]
        signal_date = date.fromisoformat(d) if d else all_sigs[0]["signal_date"]
        rows_data = self.store.list_signals(signal_date)
        rows = "".join(
            f"<tr><td>{_esc(s.get('ticker') or s['security_id'])}</td>"
            f"<td>{_esc(s['direction'])}</td>"
            f"<td><a href='/patterns/{_esc(s['pattern_id'])}'>{_esc(s['pattern_id'])}</a> v{_esc(s['pattern_version'])}</td>"
            f"<td>{_esc(s.get('hit_rate'))}</td>"
            f"<td>{_esc(s.get('hit_rate_success_rule'))}</td>"
            f"<td>{_esc(s.get('median_outcome'))}</td>"
            f"<td>{_esc(s.get('typical_loss_when_wrong'))}</td></tr>"
            for s in rows_data
        )
        body = f"""
        <h2>Signals for {_esc(signal_date)}</h2>
        <table>
          <tr><th>Ticker</th><th>Dir</th><th>Pattern</th><th>Hit rate</th>
          <th>Success rule</th><th>Median</th><th>Typical loss</th></tr>
          {rows}
        </table>
        """
        return _layout("Today's Signals", body, flash=flash)

    def render_candidates(self, qs: dict, *, flash: str | None = None) -> str:
        all_sigs = self.store.list_signals()
        if not all_sigs:
            return _layout(
                "Candidates",
                '<div class="empty"><p>No candidates. Research labels only — no broker/order actions.</p></div>',
                flash=flash,
            )
        d = (qs.get("date") or [None])[0]
        signal_date = date.fromisoformat(d) if d else all_sigs[0]["signal_date"]
        raw = self.store.list_signals(signal_date)
        signals = [
            DailyPatternSignal(
                signal_date=s["signal_date"],
                security_id=s["security_id"],
                ticker=s.get("ticker"),
                pattern_id=s["pattern_id"],
                pattern_version=s["pattern_version"],
                direction=s["direction"],
                hit_rate=s.get("hit_rate"),
                hit_rate_success_rule=s.get("hit_rate_success_rule"),
                median_outcome=s.get("median_outcome"),
            )
            for s in raw
        ]
        cands = self.aggregator.aggregate(signals)
        rows = "".join(
            f"<tr><td>{_esc(c.ticker or c.security_id)}</td><td>{_esc(c.label)}</td>"
            f"<td>{c.bullish_pattern_count}/{c.bearish_pattern_count}</td>"
            f"<td><pre>{_esc(json.dumps(c.contributing_patterns, indent=2))}</pre></td></tr>"
            for c in cands
        )
        body = f"""
        <p class="muted">BUY_CANDIDATE / SELL_CANDIDATE / CONFLICTING / WATCH are research labels, not orders.</p>
        <table>
          <tr><th>Ticker</th><th>Label</th><th>Bull/Bear counts</th><th>Evidence</th></tr>
          {rows}
        </table>
        """
        return _layout("Candidate List", body, flash=flash)


def run_dashboard(
    store: PatternResearchStore,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    persistence_enabled: bool = False,
    signal_persistence_enabled: bool = False,
    source_status: SourceStatusProvider | None = None,
) -> None:
    app = DashboardApp(
        store,
        persistence_enabled=persistence_enabled,
        signal_persistence_enabled=signal_persistence_enabled,
        host=host,
        port=port,
        source_status=source_status,
    )
    app.serve_forever()
