"""Build docs/real2sim_camera_calibration_en.html from the English notes (markdown), with the
same styles as the Chinese page.

    python docs/build_en_html.py        (needs: pip install markdown)

Images become <figure> with their alt text as the caption, tables are wrapped for horizontal
scrolling, every "## " heading starts a section, and a table of contents is added.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "real2sim_camera_calibration_en.md"
OUT = HERE / "real2sim_camera_calibration_en.html"
ZH = HERE / "real2sim_camera_calibration.html"


def main() -> None:
    zh = ZH.read_text(encoding="utf-8")
    style = zh[zh.index('<meta name="claude:viewport-fit-sides"'):zh.index("</style>") + len("</style>")]
    style = style.replace('"Noto Sans SC", "Microsoft YaHei", "PingFang SC", system-ui', 'system-ui, "Segoe UI", Roboto')
    script = zh[zh.index("<script>"):zh.index("</script>") + len("</script>")]

    text = SRC.read_text(encoding="utf-8")
    title_line, body_md = text.split("\n", 1)
    title = title_line.lstrip("# ").strip()
    intro_md, rest_md = body_md.split("\n---\n", 1)
    body = markdown.markdown(rest_md, extensions=["tables", "fenced_code"])
    intro = markdown.markdown(intro_md, extensions=["tables", "fenced_code"])

    def fig(m):
        alt, src = html.unescape(m.group(1)), m.group(2)
        label, _, cap = alt.partition(":")
        cap_html = f"<b>{html.escape(label)}</b>{html.escape(':' + cap) if cap else ''}"
        return (f'<figure><img src="{src}" alt="{html.escape(alt)}" tabindex="0" loading="lazy">'
                f"<figcaption>{cap_html}</figcaption></figure>")

    def figs(s):
        s = re.sub(r'<p>((?:<img[^>]*>\s*)+)</p>', lambda m: m.group(1), s)
        s = re.sub(r'<img alt="([^"]*)" src="([^"]*)"\s*/?>', fig, s)
        return s.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")

    body, intro = figs(body), figs(intro)
    # sections
    parts = re.split(r"<h2>(.*?)</h2>", body)
    sections, toc = [], []
    for k in range(1, len(parts), 2):
        head, content = parts[k], parts[k + 1].replace("<hr />", "")
        m = re.match(r"(\d+)\.\s*(.*)", head)
        no, name = (m.group(1), m.group(2)) if m else ("", head)
        sid = f"s{no or k}"
        toc.append(f'<li><a href="#{sid}">{no + ". " if no else ""}{name}</a></li>')
        sections.append(f'<section id="{sid}">\n  <h2><span class="no">{no}</span>{name}</h2>\n{content}\n</section>')

    page = f"""<title>ORCA Real2Sim Notes</title>
{style}

<div class="wrap">

<header style="display:grid;gap:14px">
  <div class="eyebrow">SEP 730 · ORCA dexterous hand · real2sim · 2026-11-18 → 11-21</div>
  <h1>{html.escape(title)}</h1>
{intro}
</header>

<nav class="toc" aria-label="Contents"><ol>{''.join(toc)}</ol></nav>

{chr(10).join(sections)}

<footer>Click any image to enlarge. Generated from <code>docs/real2sim_camera_calibration_en.md</code> by
<code>docs/build_en_html.py</code>; figures in <code>docs/figures/real2sim/</code>. Chinese version:
<code>docs/real2sim_camera_calibration.html</code>.</footer>
</div>

<div id="lb" hidden role="dialog" aria-label="Enlarged image"><img alt=""></div>
{script}
"""
    OUT.write_text(page, encoding="utf-8")
    print("wrote", OUT, f"({len(sections)} sections, {page.count('<figure>')} figures)")


if __name__ == "__main__":
    main()
