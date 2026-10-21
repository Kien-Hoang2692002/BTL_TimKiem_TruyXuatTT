"""Chuyển kich_ban_thuyet_trinh.md và cau_hoi_QA.md thành 2 trang HTML tự chứa.

Cách chạy (cần package `markdown`):
    python tools/build_docs.py .
"""
import re
import sys

import markdown

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

CSS = r"""
:root{
  --bg:#f6f7fb; --surface:#ffffff; --fg:#161a2b; --mute:#5d6584; --line:#e3e6f0;
  --accent:#6a3df0; --accent2:#0a8fa8; --pink:#d6337f; --quote:#f1effd;
  --font:"Be Vietnam Pro",system-ui,sans-serif; --mono:"JetBrains Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0a0c16; --surface:#121528; --fg:#e7eaf7; --mute:#97a0c4; --line:#242a45;
  --accent:#a585ff; --accent2:#3ee6ff; --pink:#ff6fb5; --quote:#191c34; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0a0c16; --surface:#121528; --fg:#e7eaf7; --mute:#97a0c4; --line:#242a45;
  --accent:#a585ff; --accent2:#3ee6ff; --pink:#ff6fb5; --quote:#191c34; color-scheme:dark}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:80px}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.7 var(--font);-webkit-font-smoothing:antialiased}
.top{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 85%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.top .in{max-width:1180px;margin:0 auto;padding:12px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.brand{font-weight:800;letter-spacing:-.01em}
.brand span{background:linear-gradient(100deg,var(--accent2),var(--accent),var(--pink));-webkit-background-clip:text;background-clip:text;color:transparent}
.nav{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap}
.nav a{font:600 13px var(--font);color:var(--mute);text-decoration:none;padding:6px 12px;border-radius:99px;border:1px solid var(--line)}
.nav a:hover,.nav a:focus-visible{color:var(--fg);border-color:var(--accent);outline:none}
.nav a.on{background:var(--accent);border-color:var(--accent);color:#fff}
#bar{position:fixed;top:0;left:0;height:3px;width:0;background:linear-gradient(90deg,var(--accent2),var(--accent),var(--pink));z-index:6}
.wrap{max-width:1180px;margin:0 auto;padding-inline:16px;padding-block:28px 80px;display:grid;grid-template-columns:250px minmax(0,1fr);gap:40px}
aside{position:sticky;top:76px;align-self:start;max-height:calc(100vh - 96px);overflow:auto;font-size:13.5px}
aside .lbl{font:600 11px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin-bottom:8px}
aside a{display:block;padding:5px 10px;border-left:2px solid var(--line);color:var(--mute);text-decoration:none;line-height:1.4}
aside a:hover{color:var(--fg)}
aside a.on{border-color:var(--accent);color:var(--fg);font-weight:600}
main{min-width:0}
h1{font-size:clamp(28px,4vw,42px);line-height:1.15;letter-spacing:-.02em;margin:0 0 10px;text-wrap:balance}
h2{font-size:22px;line-height:1.3;margin:0 0 12px;text-wrap:balance}
h3{font-size:17px;margin:18px 0 6px}
p{margin:8px 0}
a{color:var(--accent)}
ul,ol{padding-left:22px;margin:8px 0}
li{margin:4px 0}
code{font:13.5px var(--mono);background:var(--quote);padding:1px 6px;border-radius:6px}
hr{display:none}
blockquote{margin:12px 0;padding:12px 16px;background:var(--quote);border-left:3px solid var(--accent);border-radius:0 12px 12px 0}
blockquote p{margin:6px 0}
.intro{color:var(--mute)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin:18px 0;transition:border-color .2s}
.card:target{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 20%,transparent)}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--mute);font-size:12.5px;text-transform:uppercase;letter-spacing:.05em}
.chip{display:inline-block;font:600 12px var(--mono);padding:3px 10px;border-radius:99px;background:color-mix(in srgb,var(--accent2) 15%,transparent);color:var(--accent2);margin-left:8px;vertical-align:middle}
.num{font:600 12px var(--mono);color:var(--accent);letter-spacing:.1em;display:block;margin-bottom:2px}
.tools{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 4px}
.tools button{font:600 13px var(--font);background:var(--surface);color:var(--fg);border:1px solid var(--line);border-radius:99px;padding:7px 14px;cursor:pointer}
.tools button:hover,.tools button:focus-visible{border-color:var(--accent);outline:none}
.tools button.on{background:var(--accent);border-color:var(--accent);color:#fff}
details{border:1px solid var(--line);border-radius:14px;background:var(--surface);margin:10px 0;overflow:hidden}
details[hidden]{display:none}
summary{cursor:pointer;list-style:none;padding:14px 18px;font-weight:600;display:flex;gap:12px;align-items:flex-start}
summary::-webkit-details-marker{display:none}
summary::before{content:"+";flex:none;width:22px;height:22px;border-radius:7px;display:grid;place-items:center;background:var(--quote);color:var(--accent);font:700 15px var(--mono);margin-top:2px}
details[open] summary::before{content:"−"}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.ans{padding:0 18px 14px 52px;border-top:1px dashed var(--line)}
.sec{margin-top:34px}
.sec>h2{display:flex;align-items:center;gap:10px}
.count{font:600 12px var(--mono);color:var(--mute)}
@media (max-width:860px){.wrap{grid-template-columns:1fr}aside{display:none}.ans{padding-left:18px}}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}
"""

HEAD = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;800&family=JetBrains+Mono:wght@600&display=swap" rel="stylesheet">
<style>{css}</style></head><body>
<div id="bar"></div>
<header class="top"><div class="in">
 <div class="brand"><span>SLQ</span> · Nhóm báo cáo</div>
 <nav class="nav">
  <a href="bao_cao_SLQ.html">Slide</a>
  <a href="kich_ban_thuyet_trinh.html" {on1}>Kịch bản</a>
  <a href="cau_hoi_QA.html" {on2}>Hỏi đáp</a>
 </nav>
</div></header>
<div class="wrap"><aside><div class="lbl">Mục lục</div>{toc}</aside><main>
"""

JS = """
<script>
(() => {
  const bar = document.getElementById('bar');
  const links = [...document.querySelectorAll('aside a')];
  const targets = links.map(a => document.getElementById(a.hash.slice(1))).filter(Boolean);
  function onScroll() {
    const h = document.documentElement;
    bar.style.width = (h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight) * 100) + '%';
    let cur = targets[0];
    for (const t of targets) if (t.getBoundingClientRect().top < 120) cur = t;
    links.forEach(a => a.classList.toggle('on', cur && a.hash === '#' + cur.id));
  }
  addEventListener('scroll', onScroll, { passive: true }); onScroll();
  const all = [...document.querySelectorAll('details')];
  if (!all.length) return;
  const btns = [...document.querySelectorAll('[data-lv]')].filter(b => b.tagName === 'BUTTON');
  btns.forEach(b => b.onclick = () => {
    btns.forEach(x => x.classList.toggle('on', x === b));
    all.forEach(d => d.hidden = b.dataset.lv !== 'all' && d.dataset.lv !== b.dataset.lv);
  });
  document.getElementById('openAll').onclick = () => all.forEach(d => d.open = true);
  document.getElementById('closeAll').onclick = () => all.forEach(d => d.open = false);
  if (location.hash) { const d = document.getElementById(location.hash.slice(1)); if (d && d.tagName === 'DETAILS') d.open = true; }
})();
</script></main></div></body></html>
"""


def md(text):
    # Markdown cần dòng trống trước danh sách; thêm vào nếu dòng trước là văn bản thường
    text = re.sub(r"\n(?!\s*(?:[-*]|\d+\.)\s)([^\n]+)\n(?=(?:[-*]|\d+\.)\s)", "\n\\1\n\n", "\n" + text)
    return markdown.markdown(text, extensions=["tables", "sane_lists"])


def wrap_tables(html):
    return html.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")


def split_doc(path):
    src = open(path).read()
    head, *secs = re.split(r"\n---\n", src)
    title_md, _, intro_md = head.partition("\n")
    return title_md.lstrip("# "), intro_md, secs


def build_script():
    title, intro, secs = split_doc(f"{ROOT}/kich_ban_thuyet_trinh.md")
    toc, body = [], [f'<h1>{title}</h1><div class="intro">{md(intro)}</div>']
    for i, sec in enumerate(secs):
        sec = sec.strip()
        m = re.match(r"## (.+)", sec)
        if not m:
            continue
        h, rest = m.group(1), sec[m.end():]
        t = re.search(r"\*\*Thời gian:\*\*\s*([^\n]+)", rest)
        chip = f'<span class="chip">⏱ {t.group(1)}</span>' if t else ""
        rest = re.sub(r"\*\*Thời gian:\*\*\s*[^\n]+\n?", "", rest, count=1)
        sm = re.match(r"Slide (\d+): (.+)", h)
        sid = f"s{sm.group(1)}" if sm else f"p{i}"
        label = f'<span class="num">SLIDE {sm.group(1)}</span>{sm.group(2)}' if sm else h
        toc.append(f'<a href="#{sid}">{(sm.group(1) + ". " + sm.group(2)) if sm else h}</a>')
        body.append(f'<section class="card" id="{sid}"><h2>{label}{chip}</h2>{wrap_tables(md(rest))}</section>')
    html = HEAD.format(title="Kịch bản thuyết trình SLQ", css=CSS, on1='class="on"', on2="", toc="".join(toc))
    open(f"{ROOT}/kich_ban_thuyet_trinh.html", "w").write(html + "\n".join(body) + JS)


def build_qa():
    title, intro, secs = split_doc(f"{ROOT}/cau_hoi_QA.md")
    toc = []
    body = [f'<h1>{title}</h1><div class="intro">{md(intro)}</div>',
            '<div class="tools"><button data-lv="all" class="on">Tất cả</button><button data-lv="1">⭐ Cơ bản</button>'
            '<button data-lv="2">⭐⭐ Trung bình</button><button data-lv="3">⭐⭐⭐ Khó</button>'
            '<button id="openAll">Mở hết đáp án</button><button id="closeAll">Đóng hết</button></div>']
    for i, sec in enumerate(secs):
        sec = sec.strip()
        m = re.match(r"## (.+)", sec)
        if not m:
            continue
        h, rest = m.group(1), sec[m.end():]
        sid = f"g{i}"
        toc.append(f'<a href="#{sid}">{h}</a>')
        out, n = [], 0
        for q in re.split(r"\n(?=### )", rest):
            q = q.strip()
            qm = re.match(r"### (.+)", q)
            if not qm:
                if q:
                    out.append(wrap_tables(md(q)))
                continue
            n += 1
            qt = qm.group(1)
            num = re.match(r"(\d+)\.", qt)
            qid = f"q{num.group(1)}" if num else f"{sid}q{n}"
            qhtml = md(qt).strip().removeprefix("<p>").removesuffix("</p>")
            out.append(f'<details id="{qid}" data-lv="{qt.count("⭐")}"><summary><span>{qhtml}</span></summary>'
                       f'<div class="ans">{wrap_tables(md(q[qm.end():]))}</div></details>')
        cnt = f'<span class="count">{n} câu</span>' if n else ""
        body.append(f'<section class="sec" id="{sid}"><h2>{h}{cnt}</h2>{"".join(out)}</section>')
    html = HEAD.format(title="Hỏi đáp SLQ", css=CSS, on1="", on2='class="on"', toc="".join(toc))
    open(f"{ROOT}/cau_hoi_QA.html", "w").write(html + "\n".join(body) + JS)


if __name__ == "__main__":
    build_script()
    build_qa()
    print("ok")
