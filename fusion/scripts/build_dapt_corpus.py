"""Build a DAPT corpus from the dataset's own pdf_url sources (full ESG reports).

Pipeline: download report PDFs -> extract text (pymupdf) -> split into lines ->
clean (drop TOC/page-numbers/table noise, require CJK content) -> dedup -> write
one paragraph per line. Optionally append the 4000 task snippets (--add-task) to
make it DAPT+TAPT.

  python scripts/build_dapt_corpus.py --urls /tmp/pdf_ok.txt --out data/dapt/dapt_corpus.txt
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from urllib.parse import urlparse

import fitz  # pymupdf

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[一-鿿]")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
BLOB = "hsxn1sjvkgtdpixe.public.blob.vercel-storage.com"  # competition blob (blocked) — skip
DATA_FILES = ["data/raw/vpesg_4k_train_1000.json", "data/raw/vpesg4k_val_1000.json",
              "data/raw/vpesg4k_test_2000.json"]


def derive_urls():
    """Pull the non-blob (official-site) pdf_urls straight from the dataset, dedup."""
    urls = set()
    for f in DATA_FILES:
        for r in json.loads((ROOT / f).read_text()):
            u = r.get("pdf_url") or ""
            if u and urlparse(u).netloc != BLOB:
                urls.add(u)
    return sorted(urls)


def downloadable(urls):
    """Keep only URLs that actually serve application/pdf (skip HTML landing / 403)."""
    ok = []
    for u in urls:
        try:
            ct = subprocess.run(["curl", "-s", "-A", UA, "-L", "-r", "0-1023",
                                 "-o", os.devnull, "-w", "%{content_type}", "--max-time", "20", u],
                                capture_output=True, text=True).stdout
        except Exception:
            ct = ""
        if ct.startswith("application/pdf"):
            ok.append(u)
    return ok


def download(urls, pdf_dir):
    pdf_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, u in enumerate(urls):
        out = pdf_dir / f"report_{i:03d}.pdf"
        if out.exists() and out.stat().st_size > 100_000:
            paths.append(out); continue
        print(f"[{i+1}/{len(urls)}] downloading {u[:70]}", flush=True)
        rc = subprocess.run(["curl", "-s", "-A", UA, "-L", "--max-time", "180",
                             "-o", str(out), u]).returncode
        if rc == 0 and out.exists() and out.stat().st_size > 100_000:
            paths.append(out)
        else:
            print(f"   failed (rc={rc})", flush=True)
    return paths


def clean_lines(text, min_chars, cjk_frac):
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) < min_chars:
            continue
        cjk = len(CJK.findall(ln))
        if cjk / len(ln) < cjk_frac:          # mostly non-CJK (tables/numbers/headers)
            continue
        digits = sum(c.isdigit() for c in ln)
        if digits / len(ln) > 0.4:            # number-heavy table rows
            continue
        out.append(ln)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default=None, help="optional URL list file; if omitted, derive from dataset pdf_url")
    ap.add_argument("--pdf-dir", default=str(ROOT / "data/dapt/pdfs"))
    ap.add_argument("--out", default=str(ROOT / "data/dapt/dapt_corpus_full.txt"))
    ap.add_argument("--min-chars", type=int, default=24)
    ap.add_argument("--cjk-frac", type=float, default=0.35)
    ap.add_argument("--max-pages", type=int, default=400, help="cap pages/report")
    ap.add_argument("--add-task", action="store_true", help="append the 4000 task snippets (DAPT+TAPT)")
    a = ap.parse_args()

    if a.urls:
        urls = [l.strip() for l in Path(a.urls).read_text().splitlines() if l.strip()]
    else:
        print("deriving report URLs from dataset pdf_url (non-blob) ...", flush=True)
        cand = derive_urls()
        urls = downloadable(cand)
        print(f"  {len(cand)} non-blob urls -> {len(urls)} serve application/pdf", flush=True)
    pdfs = download(urls, Path(a.pdf_dir))
    print(f"downloaded {len(pdfs)} PDFs", flush=True)

    seen, corpus = set(), []
    n_pages = 0
    for p in pdfs:
        try:
            doc = fitz.open(p)
        except Exception as e:
            print(f"  open fail {p.name}: {e}", flush=True); continue
        for pi, page in enumerate(doc):
            if pi >= a.max_pages:
                break
            n_pages += 1
            for ln in clean_lines(page.get_text(), a.min_chars, a.cjk_frac):
                if ln not in seen:
                    seen.add(ln); corpus.append(ln)
        doc.close()

    if a.add_task:
        for f in ["data/raw/vpesg_4k_train_1000.json", "data/raw/vpesg4k_val_1000.json",
                  "data/raw/vpesg4k_test_2000.json"]:
            for r in json.loads((ROOT / f).read_text()):
                t = re.sub(r"\s+", " ", (r.get("data") or "")).strip()
                if t and t not in seen:
                    seen.add(t); corpus.append(t)

    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(corpus) + "\n", encoding="utf-8")
    chars = sum(len(x) for x in corpus)
    print(f">> pages parsed: {n_pages} | corpus lines: {len(corpus)} | "
          f"{chars:,} chars (~{chars//2:,} CJK tokens) | add_task={a.add_task}")
    print(f">> wrote {outp}")


if __name__ == "__main__":
    main()
