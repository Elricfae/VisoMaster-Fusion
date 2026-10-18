import os
import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# --- PyTorch VRAM Optimization ---
# MUST be set BEFORE importing torch (which occurs during _run_app).
# Instructs the caching allocator to release unused memory segments back to the CUDA driver.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import torch
torch.set_grad_enabled(False)


def _write_crash_log(exc: BaseException) -> Path:
    """Persist a full traceback to disk so the diagnostic survives even if the
    console window closes before the user can copy it.

    Returns the path of the written log so the caller can print it.
    """
    log_dir = Path(__file__).resolve().parent / "crash_logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"crash_{stamp}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"VisoMaster crash report — {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n")
        try:
            import platform

            f.write(f"Python:   {sys.version}\n")
            f.write(f"Platform: {platform.platform()}\n")
        except Exception:
            pass
        f.write("=" * 70 + "\n\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    return log_path


def _run_app() -> None:
    """Boot the Qt app. Imports are inside the function so any startup error is
    captured by the outer try/except (otherwise a top-level import error would
    bypass the crash-log writer)."""
    from PySide6 import QtWidgets
    from app.ui import main_ui
    from app.ui.core.asset_manager import AssetManager
    from app.ui.core.proxy_style import ProxyStyle
    from app.ui.core.theme_manager import ThemeManager

    parser = argparse.ArgumentParser(description="VisoMaster")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA GPU device ID to use (default: 0)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="true_dark",
        help="Initial UI theme ID (default: true_dark)",
    )
    args, remaining = parser.parse_known_args()

    app = QtWidgets.QApplication(remaining)
    app.setStyle(ProxyStyle())

    # Initialize pure-Python asset paths ('media:' and 'glyphs:') and apply theme
    AssetManager.initialize()
    ThemeManager.instance().apply_theme(args.theme, app)

    window = main_ui.MainWindow(gpu_id=args.gpu_id)
    window.show()
    app.exec()


if __name__ == "__main__":
    try:
        _run_app()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log_path = _write_crash_log(e)
        print("\n" + "=" * 70)
        print("[FATAL] VisoMaster crashed.")
        print("  Crash log written to:")
        print(f"    {log_path}")
        print(f"  Error: {e}")
        print("=" * 70)
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
