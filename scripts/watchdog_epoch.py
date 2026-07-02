"""Epoch-level watchdog: runs fusion training one epoch at a time until completion."""
import subprocess
import sys
import time
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1].parent  # 小波/
META = PROJECT / "fer_wavelet" / "runs" / "affectnet_fusion" / "run_meta.json"
LOG = PROJECT / "fer_wavelet" / "runs" / "experiment_output.log"
CMD = [
    sys.executable,
    "fer_wavelet/scripts/run_experiments.py",
    "--dataset", "affectnet",
    "--resume",
    "--batch-size", "4",
]


def is_complete() -> bool:
    if not META.exists():
        return False
    try:
        meta: dict = json.loads(META.read_text(encoding="utf-8"))
        return "finished_unix" in meta
    except Exception:
        return False


def log(msg: str):
    line = f"[WD-{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("Watchdog started, monitoring fusion training...")

    for epoch in range(1, 41):  # max 40 epochs
        if is_complete():
            meta: dict = json.loads(META.read_text(encoding="utf-8"))
            log(f"Training COMPLETE! best_val_macro_f1={meta.get('best_val_macro_f1', 'N/A')}")
            return

        log(f"Launching epoch {epoch}/40 ...")
        ret = subprocess.run(CMD, cwd=PROJECT)

        if is_complete():
            meta: dict = json.loads(META.read_text(encoding="utf-8"))
            log(f"Training COMPLETE! best_val_macro_f1={meta.get('best_val_macro_f1', 'N/A')}")
            return

        if ret.returncode != 0:
            log(f"Process exited code={ret.returncode}, retrying in 5s...")
            time.sleep(5)
            continue

        # Check run_meta for incomplete_epoch to log progress
        if META.exists():
            try:
                meta: dict = json.loads(META.read_text(encoding="utf-8"))
                ep = meta.get("incomplete_epoch", epoch)
                bf = meta.get("best_val_macro_f1", "N/A")
                log(f"Epoch {ep} done, best_val_f1={bf}")
            except Exception:
                pass

        time.sleep(2)

    log("Watchdog: all epochs dispatched.")
    if is_complete():
        meta: dict = json.loads(META.read_text(encoding="utf-8"))
        log(f"Final best_val_macro_f1={meta.get('best_val_macro_f1', 'N/A')}")
    else:
        log("WARNING: Training did not complete within 40 epochs.")


if __name__ == "__main__":
    main()
