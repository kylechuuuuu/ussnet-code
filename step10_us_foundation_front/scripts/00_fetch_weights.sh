#!/usr/bin/env bash
# step10_us_foundation_front/scripts/00_fetch_weights.sh
#
# Robust weight fetch for the ultrasound foundation models.
#
# Two failure modes found on this host (2026-10-15):
#   (1) huggingface_hub 1.28 + hf-xet talks to cas-bridge.xethub.hf.co, which is
#       BLOCKED here (403 from this IP) -> every hf_hub_download died with
#       httpx.ConnectTimeout.  Fix: plain `curl -L` against hf-mirror.com
#       (verified: 206/200 with correct Content-Length).
#   (2) a bare `curl -sL` on the CAMMA S3 truncated at 12 MB of 381 MB without a
#       non-zero exit.  Fix: --retry + `-C -` resume + size verification loop.
#
# Every download is verified against the expected byte count; a mismatch is
# retried (resumed), never silently accepted.
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$D/weights"; mkdir -p "$W"; cd "$W"

dl () {                       # dl <expected_bytes> <out> <url> [max_attempts]
    local want="$1" out="$2" url="$3" tries="${4:-8}" i=0 have=0
    for i in $(seq 1 "$tries"); do
        have=$(stat -c %s "$out" 2>/dev/null || echo 0)
        if [ "$have" = "$want" ]; then
            echo "[ok]   $out  $have bytes  (== expected)"; return 0
        fi
        echo "[dl]   attempt $i: $out  have=$have  want=$want"
        curl -L -C - --retry 5 --retry-delay 5 --retry-all-errors \
             --connect-timeout 20 --max-time 3600 -sS -o "$out" "$url" || true
    done
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    echo "[FAIL] $out  $have != $want"; return 1
}

# --- exact sizes straight from the HF mirror API (no guessing) ---------------
api_size () {                 # api_size <repo> <rfilename>
    curl -s --max-time 30 "https://hf-mirror.com/api/models/$1?blobs=true" \
      | /home/ta/anaconda3/envs/zk/bin/python -c "
import sys, json
want = '$2'
try:
    j = json.load(sys.stdin)
except Exception:
    print(0); raise SystemExit
print(next((s['size'] for s in j.get('siblings', []) if s.get('rfilename') == want), 0))
"
}

S_ULTRAFEDFM=$(api_size fmassimo/UltraFedFM checkpoint.pth)
S_FETALCLIP=$(api_size numansaeed/fetalclip-model FetalCLIP_weights.pt)
S_ULTRASAM3=$(api_size zhuqh19/UltraSAM3 UltraSAM3.pt)
echo "[sizes] UltraFedFM=$S_ULTRAFEDFM FetalCLIP=$S_FETALCLIP UltraSAM3=$S_ULTRASAM3"

# --- 1) UltraSam (CAMMA S3, 381 MB) ----------------------------------------
dl 381156023 UltraSam.pth \
   "https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth"

# --- 2) UltraFedFM (federated multi-centre US FM, 1.34 GB) ------------------
dl "$S_ULTRAFEDFM" UltraFedFM_checkpoint.pth \
   "https://hf-mirror.com/fmassimo/UltraFedFM/resolve/main/checkpoint.pth"

# --- 3) FetalCLIP (fetal US vision-language) --------------------------------
dl "$S_FETALCLIP" FetalCLIP_weights.pt \
   "https://hf-mirror.com/numansaeed/fetalclip-model/resolve/main/FetalCLIP_weights.pt"

# --- 4) UltraSAM3 (SAM3-based US segmentation FM incl. prostate MicroSeg/RegPro)
if [ "${SKIP_ULTRASAM3:-0}" != "1" ]; then
    dl "$S_ULTRASAM3" UltraSAM3.pt \
       "https://hf-mirror.com/zhuqh19/UltraSAM3/resolve/main/UltraSAM3.pt"
fi

echo "[dl] final listing:"
ls -l "$W"
