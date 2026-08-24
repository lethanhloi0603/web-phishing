import argparse
import csv
import html
import json
import shutil
import threading
import webbrowser
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "data" / "llm_phish_audit_ranked.csv"
DEFAULT_LLM = ROOT / "data" / "llm_phish.csv"
CODEBASE_ROOT = ROOT / "external" / "assessingSpearPhishing"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def local_stylesheet(source_file, href):
    """Resolve a stylesheet beside the reviewed page without escaping the dataset root."""
    if not source_file or not href or any(char in href for char in ("?", "#")):
        return ""
    source_file = Path(source_file).resolve()
    dataset_root = CODEBASE_ROOT.resolve()
    relative = href.lstrip("/\\")
    candidates = [source_file.parent / relative]
    if not href.startswith(("/", "\\")):
        candidates.insert(0, source_file.parent / href)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(dataset_root)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and resolved.suffix.lower() == ".css" and resolved.stat().st_size <= 1_000_000:
            return resolved.read_text(encoding="utf-8", errors="replace")
    return ""


def safe_preview(raw_html, source_file=None):
    """Keep visual markup while preventing scripts, navigation, and form submission."""
    soup = BeautifulSoup(str(raw_html or ""), "html.parser")
    for link in list(soup.find_all("link")):
        rel = " ".join(link.get("rel", [])).lower()
        if "stylesheet" not in rel:
            link.decompose()
            continue
        css = local_stylesheet(source_file, str(link.get("href", "")))
        if css:
            style = soup.new_tag("style")
            style.string = css
            link.replace_with(style)
        else:
            link.decompose()
    for tag in soup.find_all(["script", "iframe", "object", "embed", "base"]):
        tag.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            lowered = attr.lower()
            if lowered.startswith("on") or lowered in {"srcdoc", "formaction"}:
                del tag.attrs[attr]
        if tag.name == "form":
            tag.attrs["action"] = ""
            tag.attrs["method"] = "get"
        for attr in ("src", "href", "action"):
            value = str(tag.attrs.get(attr, "")).strip().lower()
            if value.startswith(("http:", "https:", "//", "javascript:", "data:text/html")):
                tag.attrs[attr] = ""
    return str(soup)


def source_path(row):
    relative = str(row.get("file_path", "")).strip().replace("\\", "/")
    if not relative:
        return None
    try:
        resolved = (CODEBASE_ROOT / relative).resolve()
        resolved.relative_to(CODEBASE_ROOT.resolve())
        return resolved if resolved.is_file() else None
    except (OSError, ValueError):
        return None


class ReviewStore:
    def __init__(
        self, audit_path, llm_path, only_ai_validated=True, reviewer_id=""
    ):
        self.audit_path = audit_path
        self.llm_path = llm_path
        self.lock = threading.Lock()
        self.audit_rows = read_csv(audit_path)
        self.reviewer_id = reviewer_id.strip()
        if self.reviewer_id and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.reviewer_id):
            raise ValueError("reviewer_id must start with a letter and contain only letters, numbers, or _")
        self.reviewer_columns = {}
        if self.reviewer_id:
            self.reviewer_columns = {
                "label": f"{self.reviewer_id}_label",
                "reason": f"{self.reviewer_id}_reason",
                "reviewed_at": f"{self.reviewer_id}_reviewed_at",
            }
            for row in self.audit_rows:
                for column in self.reviewer_columns.values():
                    row.setdefault(column, "")
        html_by_id = {row.get("sample_id", ""): row.get("html", "") for row in read_csv(llm_path)}
        candidates = self.audit_rows
        if only_ai_validated:
            candidates = [row for row in candidates if row.get("ai_validated", "").strip() == "1"]
        self.rows = []
        for row in candidates:
            item = dict(row)
            item["html"] = html_by_id.get(row.get("sample_id", ""), "")
            self.rows.append(item)
        self.backup_created = False

    def public_row(self, index):
        row = self.rows[index]
        fields = [
            "sample_id", "page_title", "generator", "generator_family", "brand",
            "triage_score", "triage_level", "triage_reasons", "visible_text_preview",
            "form_count", "input_count", "password_count", "file_path",
            "manual_validated", "review_status", "validation_reason",
        ]
        public = {key: row.get(key, "") for key in fields}
        if self.reviewer_columns:
            public["reviewer_id"] = self.reviewer_id
            public["your_label"] = row.get(self.reviewer_columns["label"], "")
            public["your_reason"] = row.get(self.reviewer_columns["reason"], "")
        return public

    def save_decision(self, sample_id, decision, reason):
        mapping = {
            "phishing": ("1", "human_accepted"),
            "not_phishing": ("0", "human_rejected"),
            "uncertain": ("uncertain", "human_uncertain"),
        }
        if decision not in mapping:
            raise ValueError("Invalid decision")
        manual, status = mapping[decision]
        with self.lock:
            if not self.backup_created:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = self.audit_path.with_name(f"{self.audit_path.stem}.backup_{stamp}.csv")
                shutil.copy2(self.audit_path, backup)
                self.backup_created = True
            found = False
            for row in self.audit_rows:
                if row.get("sample_id") == sample_id:
                    if self.reviewer_columns:
                        row[self.reviewer_columns["label"]] = manual
                        row[self.reviewer_columns["reason"]] = reason.strip() or status
                        row[self.reviewer_columns["reviewed_at"]] = datetime.now().isoformat(timespec="seconds")
                    else:
                        row["manual_validated"] = "1" if manual == "1" else "0"
                        row["review_status"] = status
                        row["validated_by"] = "human_reviewer"
                        row["validation_status"] = "accepted" if manual == "1" else status
                        row["validation_reason"] = reason.strip() or status
                    found = True
                    break
            if not found:
                raise ValueError("Unknown sample_id")
            fieldnames = list(self.audit_rows[0].keys())
            temp = self.audit_path.with_suffix(self.audit_path.suffix + ".tmp")
            with temp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.audit_rows)
            temp.replace(self.audit_path)
            for row in self.rows:
                if row.get("sample_id") == sample_id:
                    if self.reviewer_columns:
                        row.update({
                            self.reviewer_columns["label"]: manual,
                            self.reviewer_columns["reason"]: reason.strip() or status,
                        })
                    else:
                        row.update({
                            "manual_validated": "1" if manual == "1" else "0",
                            "review_status": status,
                            "validation_reason": reason.strip() or status,
                        })


PAGE = """<!doctype html><html><head><meta charset='utf-8'><title>LLM Phish Human Review</title>
<style>
body{font-family:Segoe UI,Arial;margin:0;background:#f4f6f8;color:#17202a}header{padding:12px 18px;background:#17202a;color:white}
.grid{display:grid;grid-template-columns:360px 1fr;gap:12px;padding:12px;height:calc(100vh - 66px);box-sizing:border-box}
.panel{background:white;border:1px solid #d8dee4;border-radius:8px;padding:14px;overflow:auto}iframe{width:100%;height:100%;border:1px solid #bbb;background:white}
button{padding:10px 12px;margin:4px;border:0;border-radius:6px;cursor:pointer}.yes{background:#16803c;color:white}.no{background:#c9362b;color:white}.maybe{background:#d99a00}.nav{background:#dfe5eb}
textarea{width:100%;height:70px;box-sizing:border-box}dt{font-weight:600;margin-top:8px}dd{margin-left:0;word-break:break-word}.warning{background:#fff3cd;padding:8px;border-radius:5px}
</style></head><body><header><b>Human review LLM phishing</b> — <span id='progress'></span></header>
<div class='grid'><section class='panel'><div class='warning'>Bản xem trước đã chặn script, gửi form và tài nguyên mạng. Không nhập dữ liệu thật.</div>
<dl id='meta'></dl><label>Lý do ngắn:</label><textarea id='reason' placeholder='Ví dụ: giả mạo thương hiệu và yêu cầu mật khẩu'></textarea>
<div><button class='yes' onclick="decide('phishing')">Phishing</button><button class='no' onclick="decide('not_phishing')">Không phishing</button><button class='maybe' onclick="decide('uncertain')">Chưa chắc</button></div>
<div><button class='nav' onclick='move(-1)'>← Trước</button><button class='nav' onclick='move(1)'>Sau →</button></div></section>
<section class='panel' style='padding:4px'><iframe id='preview' sandbox></iframe></section></div>
<script>
let index=0,total=0,current=null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function load(){const r=await fetch('/api/sample?index='+index);if(!r.ok)return;const d=await r.json();total=d.total;current=d.row;
document.getElementById('progress').textContent=`${index+1}/${total}`;document.getElementById('meta').innerHTML=Object.entries(current).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
document.getElementById('reason').value=current.validation_reason||'';document.getElementById('preview').src='/preview?index='+index;}
function move(delta){index=Math.max(0,Math.min(total-1,index+delta));load()}
async function decide(decision){const reason=document.getElementById('reason').value;const r=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample_id:current.sample_id,decision,reason})});if(!r.ok){alert(await r.text());return}if(index<total-1)index++;load()}
load();
</script></body></html>"""


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body, content_type="text/html; charset=utf-8", status=200, csp=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if csp:
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(PAGE.encode("utf-8"))
                return
            try:
                index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                if not 0 <= index < len(store.rows):
                    raise IndexError
                if parsed.path == "/api/sample":
                    payload = json.dumps({"row": store.public_row(index), "total": len(store.rows)}, ensure_ascii=False).encode("utf-8")
                    self.send_bytes(payload, "application/json; charset=utf-8")
                    return
                if parsed.path == "/preview":
                    row = store.rows[index]
                    body = safe_preview(row.get("html", ""), source_path(row)).encode("utf-8")
                    self.send_bytes(body, csp=True)
                    return
            except (ValueError, IndexError):
                pass
            self.send_bytes(b"Not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/decision":
                self.send_bytes(b"Not found", "text/plain", 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                store.save_decision(payload["sample_id"], payload["decision"], payload.get("reason", ""))
                self.send_bytes(b'{"ok":true}', "application/json")
            except Exception as exc:
                self.send_bytes(str(exc).encode("utf-8"), "text/plain", 400)

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Safe local UI for human validation of LLM HTML samples")
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--llm", default=str(DEFAULT_LLM))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--all", action="store_true", help="Review all rows, not only ai_validated=1")
    parser.add_argument(
        "--reviewer-id", default="",
        help="Independent reviewer prefix, e.g. reviewer_a; does not overwrite manual_validated",
    )
    args = parser.parse_args()
    store = ReviewStore(
        Path(args.audit).resolve(), Path(args.llm).resolve(), not args.all,
        reviewer_id=args.reviewer_id,
    )
    if not store.rows:
        raise SystemExit("No rows selected for review")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Reviewing {len(store.rows)} samples at {url}")
    if args.reviewer_id:
        print(f"Independent reviewer mode: {args.reviewer_id} (other reviewer labels are hidden)")
    print("Press Ctrl+C after finishing. Decisions are saved immediately.")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
