"""Export scripts/demo_indic.py's terminal output as a screenshot-ready
raw text file and a rendered PNG image, for the Indic evaluation pack demo.

Writes:
    E:/reports/demo/indic-demo.txt   (raw stdout, byte-for-byte)
    E:/reports/demo/indic-demo.png   (rendered dark-theme terminal image)

Render path 1 (preferred): rich.console.Console(record=True) -> export_svg()
with a custom dark TerminalTheme, then cairosvg.svg2png(scale=2.0).

Render path 2 (fallback): a direct Pillow renderer. Used automatically when
cairosvg's native cairo library isn't available -- a well-known Windows
pip-install pain point where `pip install cairosvg` succeeds but
`svg2png()` fails at runtime with ``OSError: no library called "cairo-2"
was found`` (confirmed to happen on this machine).

Installs rich/cairosvg/pillow into the *current* venv only if missing --
never touches pyproject.toml.

Usage (from the repo root):
    python scripts/export_demo.py
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("E:/reports/demo")
TXT_PATH = OUT_DIR / "indic-demo.txt"
PNG_PATH = OUT_DIR / "indic-demo.png"

# Dark VS-Code-ish palette.
BG = (0x1E, 0x1E, 0x1E)
FG = (0xD4, 0xD4, 0xD4)
GREEN = (0x4C, 0xAF, 0x50)
RED = (0xF4, 0x47, 0x47)
ACCENT = (0x4F, 0xC1, 0xFF)  # rule lines / title
HEADER = (0xDC, 0xDC, 0xAA)  # [1] [2] [3] section markers

SCALE = 2


def _ensure_installed(*packages: str) -> None:
    """pip install into this venv only, skipping anything already present."""
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing],
            check=True,
        )


def _capture_demo_lines() -> list[str]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo_indic.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    TXT_PATH.write_text(result.stdout, encoding="utf-8")
    lines = result.stdout.splitlines()
    # Trim to the bordered block: first "====" rule through the last
    # non-blank line (the closing "====" rule right after the summary).
    start = next(i for i, l in enumerate(lines) if l.strip(" ") and set(l.strip()) == {"="})
    end = max(i for i, l in enumerate(lines) if l.strip())
    return lines[start : end + 1]


def _classify_line(line: str) -> tuple[str, tuple[int, int, int]]:
    """Return (kind, color) for whole-line styling (rules/headers/title)."""
    stripped = line.strip()
    if stripped and set(stripped) == {"="}:
        return "rule", ACCENT
    if line.startswith("["):
        return "header", HEADER
    return "plain", FG


def _segments(line: str) -> list[tuple[str, tuple[int, int, int]]]:
    """Split one line into (text, color) runs; PASS/FAIL tokens get color."""
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    if stripped.startswith("PASS"):
        return [(line[: indent + 4], GREEN), (line[indent + 4 :], FG)]
    if stripped.startswith("FAIL"):
        return [(line[: indent + 4], RED), (line[indent + 4 :], FG)]
    kind, color = _classify_line(line)
    return [(line, color)]


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def render_via_rich_and_cairosvg(lines: list[str]) -> bool:
    """Preferred path. Returns False (no exception) if it can't complete,
    so the caller can fall back to Pillow.

    cairosvg's underlying cairocffi binding calls ``ctypes`` ``dlopen`` on
    the native cairo library *at import time* -- when that native library
    isn't installed (common on Windows with no system Cairo), the import
    itself raises ``OSError``, not ``ImportError``. Both are caught here,
    at both the "is it already importable" check and the "install it, then
    try again" retry, so a broken cairosvg never crashes this script --
    it just falls back to the Pillow renderer below.
    """
    try:
        _ensure_installed("rich")
        from rich.console import Console
        from rich.terminal_theme import TerminalTheme
        from rich.text import Text
    except ImportError as exc:
        print(f"[export_demo] rich unavailable ({exc}); using Pillow fallback", file=sys.stderr)
        return False

    try:
        import cairosvg
    except Exception:  # noqa: BLE001 - ImportError (missing) or OSError (native lib missing)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "cairosvg"],
                check=True,
            )
            import cairosvg
        except Exception as exc:  # noqa: BLE001
            print(
                f"[export_demo] cairosvg unusable on this machine ({type(exc).__name__}: {exc}); "
                "using Pillow fallback",
                file=sys.stderr,
            )
            return False

    max_len = max((len(l) for l in lines), default=80)
    console = Console(record=True, width=max_len, height=len(lines) + 2)

    theme = TerminalTheme(
        BG,
        FG,
        [BG, RED, GREEN, HEADER, ACCENT, ACCENT, ACCENT, FG],
        [BG, RED, GREEN, HEADER, ACCENT, ACCENT, ACCENT, (255, 255, 255)],
    )

    for line in lines:
        text = Text()
        for chunk, color in _segments(line):
            text.append(chunk, style=_rgb_to_hex(color))
        console.print(text)

    svg = console.export_svg(title="AgentEval -- Indic-Language Evaluation Pack", theme=theme)

    try:
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(PNG_PATH), scale=float(SCALE))
    except OSError as exc:
        print(f"[export_demo] cairosvg could not rasterize ({exc}); using Pillow fallback", file=sys.stderr)
        return False
    return True


def render_via_pillow(lines: list[str]) -> None:
    """Fallback path: draw directly onto a bitmap. No SVG, no cairo."""
    _ensure_installed("PIL")
    from PIL import Image, ImageDraw, ImageFont

    font_size = 16 * SCALE
    font = None
    font_bold = None
    for candidate in ("consola.ttf", "cascadiacode.ttf", "lucon.ttf", "cour.ttf"):
        path = Path(r"C:\Windows\Fonts") / candidate
        if path.is_file():
            font = ImageFont.truetype(str(path), font_size)
            break
    if font is None:
        font = ImageFont.load_default(size=font_size)
    bold_path = Path(r"C:\Windows\Fonts\consolab.ttf")
    font_bold = ImageFont.truetype(str(bold_path), font_size) if bold_path.is_file() else font

    char_w = font.getlength("M")
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * 1.35)
    pad = 24 * SCALE

    max_len = max((len(l) for l in lines), default=80)
    width = int(char_w * max_len) + pad * 2
    height = line_h * len(lines) + pad * 2

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    y = pad
    for line in lines:
        x = pad
        _kind, whole_color = _classify_line(line)
        use_bold = _kind in ("rule", "header")
        for chunk, color in _segments(line):
            draw.text((x, y), chunk, font=font_bold if use_bold else font, fill=color)
            x += font.getlength(chunk)
        y += line_h

    img.save(PNG_PATH)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = _capture_demo_lines()

    ok = render_via_rich_and_cairosvg(lines)
    if not ok:
        render_via_pillow(lines)
        print(f"[export_demo] rendered via Pillow fallback -> {PNG_PATH}")
    else:
        print(f"[export_demo] rendered via rich + cairosvg -> {PNG_PATH}")

    print(f"[export_demo] raw text -> {TXT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
