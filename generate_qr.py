from __future__ import annotations

import argparse
import re
from pathlib import Path

import qrcode


def safe_filename(text: str, max_len: int = 80) -> str:
    """
    Make a safe filename from any input string.
    Keeps letters/numbers/._- and replaces the rest with underscores.
    """
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-") or "qr"
    return text[:max_len]


def generate_qr_png(data: str, out_dir: Path, box_size: int = 10, border: int = 4) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    qr = qrcode.QRCode(
        version=None,  # auto
        error_correction=qrcode.constants.ERROR_CORRECT_M,  # good default
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    filename = f"{safe_filename(data)}.png"
    out_path = out_dir / filename
    img.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a QR code PNG from input text.")
    parser.add_argument("data", nargs="?", help="Data/text to encode (e.g., BIN-XXXX or CMP-XXXX).")
    parser.add_argument("--out", default="out_qr", help="Output directory (default: out_qr)")
    parser.add_argument("--box", type=int, default=10, help="QR box size (default: 10)")
    parser.add_argument("--border", type=int, default=4, help="QR border size (default: 4)")
    args = parser.parse_args()

    data = args.data
    if not data:
        data = input("Enter code/text to encode: ").strip()

    if not data:
        raise SystemExit("No data provided.")

    out_path = generate_qr_png(data, Path(args.out), box_size=args.box, border=args.border)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
