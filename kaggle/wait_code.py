"""
Wait until Kaggle really SERVES the code version that was just uploaded.

    python kaggle/wait_code.py

`kaggle datasets status` can say "ready" for the previous version while the new one is still being
processed, and a job started in that window silently runs old code. This compares the served
CODE_VERSION.txt with the local one instead.
"""
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
want = open(os.path.join(ROOT, "kaggle", "_build", "code", "CODE_VERSION.txt"), encoding="utf-8").read().strip()
with tempfile.TemporaryDirectory() as t:
    got = None
    for i in range(40):
        subprocess.run(["kaggle", "datasets", "download", "omarashour206/retinograde-code", "-f", "CODE_VERSION.txt",
                        "-p", t, "--force"], capture_output=True)
        p = os.path.join(t, "CODE_VERSION.txt")
        got = open(p, encoding="utf-8").read().strip() if os.path.isfile(p) else None
        if got == want:
            print(f"served: {got} (checked {i + 1}x)")
            sys.exit(0)
        time.sleep(10)
sys.exit(f"still serving {got!r}, wanted {want!r}")
