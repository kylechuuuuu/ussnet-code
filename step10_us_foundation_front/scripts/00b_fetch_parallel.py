#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/00b_fetch_parallel.py

Multi-connection (HTTP Range) downloader.  Needed because the CAMMA S3 host
serves this box at ~25 KB/s per connection (measured: a 10 MB range took 90 s)
while supporting `Accept-Ranges: bytes`, so a single stream would need ~4.2 h
for UltraSam's 381 MB.  N parallel range requests turn that into minutes.

Every chunk is fetched with its own Range request, written at its absolute
offset with os.pwrite (no shared file pointer, no locking), retried
individually, and the result is verified against the exact byte count from the
HF API / HEAD before it is accepted.

usage:
  python 00b_fetch_parallel.py --url URL --out FILE --size BYTES [--conns 32]
  python 00b_fetch_parallel.py --batch          # the step10 manifest
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = {"User-Agent": "Mozilla/5.0 (compatible; step10-fetch/1.0)"}
MIN_CHUNK = 1 << 20          # 1 MiB


def head_size(url):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_chunk(url, path, start, end, tries=6):
    """Fetch [start, end] inclusive into `path` at the right offset."""
    want = end - start + 1
    got = 0
    for attempt in range(1, tries + 1):
        headers = dict(UA, Range="bytes=%d-%d" % (start + got, end))
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                fd = os.open(path, os.O_WRONLY)
                try:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        os.pwrite(fd, buf, start + got)
                        got += len(buf)
                finally:
                    os.close(fd)
            if got >= want:
                return got
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print("   chunk %d-%d attempt %d: %s" % (start, end, attempt, e), flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError("chunk %d-%d incomplete (%d/%d)" % (start, end, got, want))


def download(url, out, size=None, conns=32):
    size = size or head_size(url)
    if os.path.exists(out) and os.path.getsize(out) == size:
        print("[ok]   %s already complete (%d bytes)" % (out, size))
        return True
    part = out + ".part"
    print("[dl]   %s  %d bytes  ~%.1f MB  conns=%d" % (out, size, size / 1e6, conns), flush=True)
    n = max(1, min(conns, size // MIN_CHUNK))
    step = size // n
    ranges = []
    pos = 0
    while pos < size:
        end = min(pos + step - 1, size - 1)
        ranges.append((pos, end))
        pos = end + 1
    # preallocate the PART file so os.pwrite lands inside it; the final name only
    # appears once every chunk is verified (a preallocated file would otherwise
    # pass the size check while still full of holes)
    with open(part, "wb") as fh:
        fh.truncate(size)

    lock = threading.Lock()
    done = [0]

    def worker(rng):
        fetch_chunk(url, part, rng[0], rng[1])
        with lock:
            done[0] += 1
            if done[0] % max(1, n // 8) == 0 or done[0] == n:
                print("   %d/%d chunks" % (done[0], n), flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(worker, r) for r in ranges]
        for f in as_completed(futs):
            f.result()
    secs = time.time() - t0
    got = os.path.getsize(part)
    ok = got == size
    print("[%s] %s  %d/%d bytes  %.1f MB/s  %.1f min"
          % ("ok" if ok else "FAIL", out, got, size, size / max(1e-9, secs) / 1e6, secs / 60), flush=True)
    if not ok:
        raise SystemExit("size mismatch for %s" % out)
    os.replace(part, out)
    return ok


def api_size(repo, fname):
    url = "https://hf-mirror.com/api/models/%s?blobs=true" % repo
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        j = json.load(r)
    for s in j.get("siblings", []):
        if s.get("rfilename") == fname:
            return int(s["size"])
    raise SystemExit("no %s in %s" % (fname, repo))


BATCH = [
    # (url, out filename, size, conns)  -- sizes verified via HEAD / HF API
    ("https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth",
     "UltraSam.pth", 381156023, 32),
    ("https://hf-mirror.com/fmassimo/UltraFedFM/resolve/main/checkpoint.pth",
     "UltraFedFM_checkpoint.pth", 1341123093, 8),
    ("https://hf-mirror.com/numansaeed/fetalclip-model/resolve/main/FetalCLIP_weights.pt",
     "FetalCLIP_weights.pt", 1710766661, 8),
    ("https://hf-mirror.com/zhuqh19/UltraSAM3/resolve/main/UltraSAM3.pt",
     "UltraSAM3.pt", 10081250438, 16),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--out")
    ap.add_argument("--size", type=int)
    ap.add_argument("--conns", type=int, default=32)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--skip-ultrasam3", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))
    os.makedirs(root, exist_ok=True)
    os.chdir(root)

    if args.batch:
        for url, name, size, conns in BATCH:
            if args.skip_ultrasam3 and name == "UltraSAM3.pt":
                continue
            download(url, name, size, conns)
        print("[dl] final listing:", flush=True)
        for f in sorted(os.listdir(".")):
            print("   %12d  %s" % (os.path.getsize(f), f), flush=True)
    else:
        if not (args.url and args.out):
            raise SystemExit("give --url/--out or --batch")
        download(args.url, args.out, args.size, args.conns)


if __name__ == "__main__":
    sys.exit(main())
