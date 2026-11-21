"""Draw the DMG window background: the app on the left, an arrow, Applications on the right.

Writes desktop/build-resources/dmg-background.tiff (1x + 2x in one file, so Finder
picks the sharp one on Retina screens) and dmg-volume.icns (the mounted volume's
icon). The icon positions it is drawn for are the `dmg.contents` entries in
desktop/package.json; change both together. Those entries also list both files a
second time, under their names on the volume (.background.tiff, .VolumeIcon.icns),
only to park them below the window: Finder shows them when hidden files are on.

    uv run python desktop/scripts/make_dmg_background.py     (macOS: needs tiffutil)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 540, 380                  # dmg.window size (points)
APP_X, APPS_X, ICON_Y = 150, 390, 150
SUPER = 4                        # draw at 4x, then downsample: smooth edges
TOP, BOTTOM = (240, 248, 241), (255, 255, 255)
ARROW = (56, 142, 60)            # the UI's --accent-strong (#388e3c)
CAPTION = (95, 107, 96)
FONTS = ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"]

RESOURCES = Path(__file__).resolve().parent.parent / "build-resources"
OUT = RESOURCES / "dmg-background.tiff"
ICON = RESOURCES / "icon.png"
VOLUME_ICON = RESOURCES / "dmg-volume.icns"


def font(size: int):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(scale: int) -> Image.Image:
    s = scale * SUPER
    img = Image.new("RGB", (W * s, H * s))
    px = img.load()
    for y in range(H * s):                       # soft vertical gradient
        t = y / (H * s - 1)
        row = tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM))
        for x in range(W * s):
            px[x, y] = row
    d = ImageDraw.Draw(img)
    # Arrow between the two icons (icons are about 100 pt wide).
    x0, x1, y = (APP_X + 72) * s, (APPS_X - 72) * s, ICON_Y * s
    thick, head_len, head_half = 7 * s, 22 * s, 16 * s
    d.rounded_rectangle([x0, y - thick // 2, x1 - head_len + 2 * s, y + thick // 2], radius=thick // 2, fill=ARROW)
    d.polygon([(x1 - head_len, y - head_half), (x1, y), (x1 - head_len, y + head_half)], fill=ARROW)
    # Caption under the icon labels, high enough to stay visible when Finder shows its
    # path bar and status bar (they cover the bottom ~90 pt of the window).
    text = "Drag VoucherVisionLocal into Applications"
    f = font(14 * s)
    width = d.textlength(text, font=f)
    d.text(((W * s - width) / 2, 238 * s), text, font=f, fill=CAPTION)
    return img.resize((W * scale, H * scale), Image.LANCZOS)


def volume_icon(tmp: Path) -> None:
    """The DMG's volume icon (.VolumeIcon.icns) from the app icon, as a standard iconset."""
    iconset = tmp / "volume.iconset"
    iconset.mkdir()
    for size in (16, 32, 128, 256, 512):
        for factor, suffix in ((1, ""), (2, "@2x")):
            subprocess.run(["sips", "-z", str(size * factor), str(size * factor), str(ICON),
                            "--out", str(iconset / f"icon_{size}x{size}{suffix}.png")],
                           check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(VOLUME_ICON)], check=True)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        one, two = Path(tmp) / "bg.png", Path(tmp) / "bg@2x.png"
        draw(1).save(one, dpi=(72, 72))
        draw(2).save(two, dpi=(144, 144))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tiffutil", "-cathidpicheck", str(one), str(two), "-out", str(OUT)], check=True)
        volume_icon(Path(tmp))
    print(f"wrote {OUT} and {VOLUME_ICON}")


if __name__ == "__main__":
    main()
