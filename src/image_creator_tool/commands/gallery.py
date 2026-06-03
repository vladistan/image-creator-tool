"""Gallery command — browse generation history visually via HTML."""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import sentry_sdk
import typer

from image_creator_tool.history import HISTORY_FILE


def gallery(
    n: int = typer.Option(100, "-n", help="Number of entries to display"),
    all_history: bool = typer.Option(False, "--all", help="Show entire history"),
    since: str | None = typer.Option(None, "--since", help="Show entries since date (YYYY-MM-DD)"),
    until: str | None = typer.Option(None, "--until", help="Show entries until date (YYYY-MM-DD)"),
    gap: int = typer.Option(30, "--gap", help="Minutes of inactivity to split sections"),
    project: str | None = typer.Option(None, "--project", help="Filter by project"),
    model: str | None = typer.Option(None, "--model", help="Filter by model"),
    preset: str | None = typer.Option(None, "--preset", help="Filter by preset"),
) -> None:
    """Open an HTML gallery of recent generations in the browser."""
    if not HISTORY_FILE.exists():
        typer.echo("No history yet.")
        return

    with HISTORY_FILE.open() as f:
        entries = [json.loads(ln) for ln in f.readlines()]

    # Date range filter
    if since:
        since_dt = datetime.fromisoformat(since)
        entries = [e for e in entries if _parse_ts(e) and _parse_ts(e) >= since_dt]  # type: ignore[operator]
    if until:
        until_dt = datetime.fromisoformat(until)
        entries = [e for e in entries if _parse_ts(e) and _parse_ts(e) <= until_dt]  # type: ignore[operator]

    # Apply filters
    if project:
        entries = [e for e in entries if e.get("project") == project]
    if model:
        entries = [e for e in entries if model in (e.get("model") or "")]
    if preset:
        entries = [e for e in entries if e.get("preset") == preset]

    if not all_history and not since:
        entries = entries[-n:]

    if not entries:
        typer.echo("No entries match filters.")
        return

    # Filter to entries with existing output files
    valid_entries = []
    for e in entries:
        path = e.get("output_path", "")
        if path and Path(path).exists():
            valid_entries.append(e)

    if not valid_entries:
        typer.echo("No images found on disk.")
        return

    with sentry_sdk.start_transaction(op="image.gallery", name="gallery") as txn:
        txn.set_data("entry_count", len(valid_entries))
        html = _build_html(valid_entries, gap_minutes=gap)
        out = Path(tempfile.mktemp(suffix=".html", prefix="image-creator-gallery-"))
        out.write_text(html)
        subprocess.run(["open", str(out)], check=False)
    typer.echo(f"Gallery opened ({len(valid_entries)} images): {out}")


def _parse_ts(entry: dict[str, Any]) -> datetime | None:
    """Parse timestamp from a history entry."""
    ts = entry.get("timestamp", "")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _build_sections(
    entries: list[dict[str, Any]], gap_minutes: int = 30
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group entries into sections based on time gaps.

    A new section starts when the gap between adjacent entries exceeds gap_minutes.
    """
    if not entries:
        return []

    gap_seconds = gap_minutes * 60
    sections: list[tuple[str, list[dict[str, Any]]]] = []
    current_group: list[dict[str, Any]] = [entries[0]]
    prev_ts = _parse_ts(entries[0])

    for entry in entries[1:]:
        curr_ts = _parse_ts(entry)
        new_section = False
        if prev_ts and curr_ts:
            delta = abs((curr_ts - prev_ts).total_seconds())
            if delta > gap_seconds:
                new_section = True
        elif not curr_ts:
            new_section = True

        if new_section:
            label = _section_label(current_group)
            sections.append((label, current_group))
            current_group = []

        current_group.append(entry)
        prev_ts = curr_ts

    if current_group:
        label = _section_label(current_group)
        sections.append((label, current_group))

    return sections


def _section_label(group: list[dict[str, Any]]) -> str:
    """Generate a human-readable label for a section."""
    ts = _parse_ts(group[0])
    if not ts:
        return "Unknown time"
    now = datetime.now()
    if ts.date() == now.date():
        return f"Today {ts.strftime('%H:%M')}"
    if (now - ts).days == 1:
        return f"Yesterday {ts.strftime('%H:%M')}"
    return ts.strftime("%b %d, %Y  %H:%M")


def _build_html(entries: list[dict[str, Any]], gap_minutes: int = 30) -> str:
    """Generate an HTML gallery with sections and lightbox."""
    sections = _build_sections(list(reversed(entries)), gap_minutes)  # newest first

    sections_html = []
    for label, group in sections:
        cards = []
        for e in group:
            path = e.get("output_path", "")
            ts = (e.get("timestamp") or "?")[:19].replace("T", " ")
            subject = e.get("subject", "")[:80]
            model_name = e.get("model", "?")
            preset_name = e.get("preset") or "-"
            platform = e.get("platform") or "-"
            edit_src = e.get("edit_source") or ""
            references = e.get("reference", []) or []
            # Build ref paths for lightbox copy buttons
            ref_paths_json = json.dumps(references) if references else "[]"
            edit_src_escaped = edit_src.replace("`", "\\`")
            info_lines = (
                f"{model_name} | preset: {preset_name} | platform: {platform}\\n{ts}\\n{subject}"
            )
            # Escape backticks for JS template literal
            info_escaped = info_lines.replace("`", "\\`")
            fname = Path(path).name

            cards.append(f"""
            <div class="card"
                 data-path="{path}"
                 data-info="{info_escaped}"
                 data-edit-src="{edit_src_escaped}"
                 data-refs='{ref_paths_json}'>
              <img src="file://{path}" loading="lazy" class="card-img" />
              <div class="meta">
                <span class="timestamp">{ts}</span>
                <span class="model">{model_name}</span>
                <span class="preset">{preset_name}</span>
              </div>
              <div class="prompt">{subject}</div>
              <div class="filename">
                <span>{fname}</span>
                <button class="fname-copy">📋</button>
              </div>
              <button class="copy-btn">📋</button>
            </div>""")

        sections_html.append(f"""
        <div class="section">
          <h2 class="section-header" onclick="toggleSection(this)">
            <span class="collapse-icon">▼</span> {label}
            <span class="section-count">({len(group)})</span>
          </h2>
          <div class="grid">{"".join(cards)}</div>
        </div>""")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Image Creator Gallery</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    background: #111; color: #eee; font-family: -apple-system, system-ui, sans-serif;
    margin: 0; padding: 20px;
  }}
  h1 {{ text-align: center; color: #fff; margin-bottom: 5px; }}
  .count {{ text-align: center; color: #666; margin-bottom: 20px; font-size: 13px; }}
  .filters {{
    text-align: center; margin-bottom: 30px;
  }}
  .filters input {{
    background: #222; border: 1px solid #444; color: #eee;
    padding: 10px 16px; border-radius: 6px; width: 400px; font-size: 14px;
  }}
  .filters input:focus {{ outline: none; border-color: #88f; }}
  .section {{ margin-bottom: 40px; }}
  .section-header {{
    color: #888; font-size: 14px; font-weight: 500; padding: 8px 12px;
    margin-bottom: 12px; border-left: 3px solid #444; cursor: pointer;
    user-select: none; transition: color 0.2s;
  }}
  .section-header:hover {{ color: #ccc; }}
  .section-count {{ color: #555; font-weight: 400; }}
  .collapse-icon {{ display: inline-block; transition: transform 0.2s; font-size: 11px; }}
  .section.collapsed .collapse-icon {{ transform: rotate(-90deg); }}
  .section.collapsed .grid {{ display: none; }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
  }}
  .card {{
    background: #1a1a1a; border-radius: 6px; overflow: hidden;
    transition: transform 0.15s, box-shadow 0.15s;
  }}
  .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
  .card img {{
    width: 100%; height: 180px; object-fit: cover; display: block; cursor: pointer;
  }}
  .meta {{
    padding: 6px 10px; font-size: 10px; color: #666; display: flex; gap: 6px;
  }}
  .meta .model {{ color: #88f; }}
  .meta .preset {{ color: #8f8; }}
  .meta .timestamp {{ color: #666; }}
  .prompt {{
    padding: 2px 10px 6px; font-size: 12px; color: #aaa;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .filename {{
    padding: 0 10px 8px; font-size: 10px; color: #555; font-family: monospace;
    display: flex; align-items: center; gap: 4px;
  }}
  .filename span {{
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0;
  }}
  .fname-copy {{
    background: none; border: none; color: #555; cursor: pointer; font-size: 10px;
    padding: 0; flex-shrink: 0; line-height: 1;
  }}
  .fname-copy:hover {{ color: #aaa; }}
  .copy-btn {{
    position: absolute; top: 6px; right: 6px; background: rgba(0,0,0,0.7);
    border: none; border-radius: 4px; padding: 4px 6px; cursor: pointer;
    font-size: 12px; opacity: 0; transition: opacity 0.2s;
  }}
  .card {{ position: relative; }}
  .card:hover .copy-btn {{ opacity: 1; }}
  /* Lightbox */
  .lightbox {{
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.95); z-index: 1000; cursor: pointer;
    flex-direction: column; justify-content: center; align-items: center;
  }}
  .lightbox.active {{ display: flex; }}
  .lightbox img {{
    max-width: 90vw; max-height: 80vh; object-fit: contain; border-radius: 4px;
  }}
  .lightbox-info {{
    color: #aaa; font-size: 13px; margin-top: 12px; text-align: center;
    white-space: pre-line; max-width: 80vw; line-height: 1.6;
  }}
  .lightbox-path {{
    color: #666; font-size: 11px; margin-top: 8px; font-family: monospace;
  }}
  .lightbox-copy {{
    background: #333; border: 1px solid #555; color: #ccc; border-radius: 3px;
    padding: 2px 8px; cursor: pointer; font-size: 11px; margin-left: 6px;
  }}
  .lightbox-copy:hover {{ background: #444; }}
  .lightbox-close {{
    position: fixed; top: 20px; right: 30px; color: #fff; font-size: 30px;
    cursor: pointer; z-index: 1001;
  }}
</style>
</head>
<body>
<h1>🐔 Image Creator Gallery</h1>
<div class="count">{len(entries)} images</div>
<div class="filters">
  <input type="text" id="search"
    placeholder="Filter by prompt, model, or preset..." />
</div>
{"".join(sections_html)}
<div class="lightbox" id="lightbox">
  <span class="lightbox-close">&times;</span>
  <img id="lightbox-img" src="" />
  <div class="lightbox-info" id="lightbox-info"></div>
  <div class="lightbox-path" id="lightbox-path"></div>
</div>
<script>
function filterCards() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.card').forEach(card => {{
    const text = card.textContent.toLowerCase();
    card.style.display = text.includes(q) ? '' : 'none';
  }});
  document.querySelectorAll('.section').forEach(sec => {{
    const visible = sec.querySelectorAll(
      '.card:not([style*="none"])'
    ).length;
    sec.style.display = visible > 0 ? '' : 'none';
  }});
}}
function toggleSection(header) {{
  header.parentElement.classList.toggle('collapsed');
}}
function copyToClip(text, btn) {{
  navigator.clipboard.writeText(text).then(() => {{
    btn.textContent = '✓';
    setTimeout(() => btn.textContent = '📋', 1000);
  }});
}}
function openLightbox(card) {{
  const path = card.dataset.path;
  const info = card.dataset.info;
  const editSrc = card.dataset.editSrc;
  const refs = JSON.parse(card.dataset.refs || '[]');
  const img = card.querySelector('img');
  document.getElementById('lightbox-img').src = img.src;
  document.getElementById('lightbox-info').textContent = info;
  const fname = path.split('/').pop();
  let pathHtml = pathEntry('📁', fname, path);
  if (editSrc) {{
    pathHtml += pathEntry('✏️', editSrc.split('/').pop(), editSrc);
  }}
  refs.forEach(r => {{
    pathHtml += pathEntry('🖼️', r.split('/').pop(), r);
  }});
  document.getElementById('lightbox-path').innerHTML = pathHtml;
  document.getElementById('lightbox').classList.add('active');
}}
function pathEntry(icon, fname, fullPath) {{
  return `<div>${{icon}} <code>${{fname}}</code> `
    + `<button class="lightbox-copy" data-clip="${{fullPath}}">`
    + `📋</button></div>`;
}}
function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('active');
}}
// Delegated event listeners
document.addEventListener('click', e => {{
  const card = e.target.closest('.card');
  if (e.target.closest('.lightbox-copy')) {{
    e.stopPropagation();
    const btn = e.target.closest('.lightbox-copy');
    copyToClip(btn.dataset.clip, btn);
  }} else if (e.target.matches('.copy-btn, .fname-copy')) {{
    e.stopPropagation();
    copyToClip(card.dataset.path, e.target);
  }} else if (e.target.matches('.card-img')) {{
    e.stopPropagation();
    openLightbox(card);
  }} else if (e.target.matches('.lightbox, .lightbox-close')) {{
    closeLightbox();
  }}
}});
document.getElementById('search').addEventListener(
  'input', filterCards
);
document.addEventListener(
  'keydown', e => {{ if (e.key === 'Escape') closeLightbox(); }}
);
</script>
</body>
</html>"""
