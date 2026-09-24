"""Full per-module test sweep: one subprocess per tests/test_*.py module.

CLAUDE.md forbids one-process `unittest discover` (hangs, cross-module state
leaks), so this runs each module alone with a timeout and records the result
as JSON, then diffs two runs to separate regressions from pre-existing failures.

    venv/bin/python tools/run_test_sweep.py run OUT.json [module ...]
    venv/bin/python tools/run_test_sweep.py diff BASE.json AFTER.json

`test_local_tts` is skipped (real playback wedges); run it through
tools/run_lean_checks.py instead. DJR3X_NO_SERVOS=1 is set so a sweep on the
robot never moves the head. Per-module timeout: SWEEP_TIMEOUT (default 300 s).

CAUTION: several test modules still write fixture rows into the REAL
assets/memory/*.db and assets/state/*.json. On a dev Mac that data is
disposable; on the robot, back those files up before a sweep.
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {"test_local_tts"}


def run(out: str, mods: list[str]) -> None:
    mods = mods or sorted(os.path.basename(p)[:-3] for p in glob.glob(f"{REPO}/tests/test_*.py"))
    mods = [m for m in mods if m not in SKIP]
    env = dict(os.environ, DJR3X_NO_SERVOS="1", PYTHONUNBUFFERED="1")
    limit = int(os.environ.get("SWEEP_TIMEOUT", "300"))
    python = os.path.join(REPO, "venv", "bin", "python")
    res = {}
    t0 = time.time()
    for i, m in enumerate(mods, 1):
        st = time.time()
        try:
            p = subprocess.run([python, "-m", "unittest", f"tests.{m}"], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=limit)
            txt = p.stdout + p.stderr
            ran = re.search(r"Ran (\d+) test", txt)
            bad = sorted(set(re.findall(r"^(FAIL|ERROR): (\S+) \(([^)]+)\)", txt, re.M)))
            status = "ok" if p.returncode == 0 else ("fail" if ran else "import_error")
            res[m] = {"status": status, "ran": int(ran.group(1)) if ran else 0,
                      "bad": [f"{k}:{c}" if c.endswith(n) else f"{k}:{c}.{n}" for k, n, c in bad],
                      "secs": round(time.time() - st, 1),
                      "tail": "" if status == "ok" else txt[-1500:]}
        except subprocess.TimeoutExpired:
            res[m] = {"status": "timeout", "ran": 0, "bad": [], "secs": limit, "tail": ""}
        r = res[m]
        print(f"[{i}/{len(mods)}] {m}: {r['status']} ran={r['ran']} bad={len(r['bad'])} {r['secs']}s", flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"DONE {len(mods)} modules in {time.time() - t0:.0f}s; "
          f"not-ok: {[m for m in res if res[m]['status'] != 'ok']}")


def diff(base_path: str, after_path: str) -> None:
    with open(base_path) as fh:
        a = json.load(fh)
    with open(after_path) as fh:
        b = json.load(fh)
    for m in sorted(set(a) | set(b)):
        if m not in b:
            print(f"GONE   {m}")
            continue
        if m not in a:
            print(f"NEW    {m}: {b[m]['status']} {b[m]['bad']}")
            continue
        sa, sb = set(a[m]["bad"]), set(b[m]["bad"])
        if b[m]["status"] in ("import_error", "timeout") and a[m]["status"] != b[m]["status"]:
            print(f"BROKE  {m}: {a[m]['status']} -> {b[m]['status']}\n{b[m]['tail'][-800:]}")
        for t in sorted(sb - sa):
            print(f"REGR   {m}: {t}")
        for t in sorted(sa - sb):
            print(f"fixed  {m}: {t}")
        if a[m]["ran"] != b[m]["ran"]:
            print(f"count  {m}: {a[m]['ran']} -> {b[m]['ran']}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 4 and sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
