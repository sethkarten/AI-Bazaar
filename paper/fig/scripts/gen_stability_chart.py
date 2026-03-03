"""
Generate stability_chart.png via matplotlib for the B2C Fix panel inset.
Run once; output cached in paper/fig/icons/
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

ICONS_DIR = Path(__file__).parent.parent / "icons"
ICONS_DIR.mkdir(exist_ok=True)
OUT = ICONS_DIR / "stability_chart.png"

if OUT.exists():
    print(f"cached: {OUT.name}")
else:
    fig, ax = plt.subplots(figsize=(512/150, 512/150), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    t = np.linspace(0, 20, 300)
    floor = 2.5
    prices = 5.0 + 2.0 * np.exp(-t / 8) + 0.3 * np.sin(t * 0.8)
    prices = np.clip(prices, floor + 0.3, None)

    # Green shaded area above floor
    ax.fill_between(t, floor, prices, color="#1E8449", alpha=0.18)

    # Dashed orange floor line
    ax.axhline(floor, color="#B7600A", ls="--", lw=2.5, dash_capstyle="round")

    # Thick green price line
    ax.plot(t, prices, color="#1E8449", lw=3.5, solid_capstyle="round")

    ax.axis("off")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(floor - 0.4, prices.max() + 0.4)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # White-transparency pass
    img = Image.open(OUT).convert("RGBA")
    data = img.getdata()
    new_data = []
    for r, g, b, a in data:
        if r > 220 and g > 220 and b > 220:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append((r, g, b, a))
    img.putdata(new_data)
    img.save(OUT, "PNG")
    print(f"saved: {OUT}")
