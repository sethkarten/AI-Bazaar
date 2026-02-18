"""
Generate custom high-quality icons via Imagen 4 for the teaser figure.
Run once; outputs are cached in paper/fig/icons/
"""
import os, io, json
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image

ICONS_DIR = Path(__file__).parent.parent / "icons"
ICONS_DIR.mkdir(exist_ok=True)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

ICON_SPECS = {
    "building_neutral": (
        "A single flat-design commercial storefront icon. "
        "Front-facing view. Simple rectangular building with a small awning and door. "
        "Solid dark charcoal color. Pure white background. "
        "Absolutely NO text, NO letters, NO numbers, NO watermarks, NO labels, NO captions, NO shadows, NO gradients. "
        "Clean minimal icon. Centered in frame with padding."
    ),
    "person_neutral": (
        "A single flat-design user profile silhouette icon. "
        "Circle head above a rounded torso shape, like a standard user avatar. "
        "Solid dark charcoal color. Pure white background. "
        "Absolutely NO text, NO letters, NO numbers, NO watermarks, NO labels, NO captions, NO shadows. "
        "Clean minimal icon. Centered in frame with padding."
    ),
    "shield_neutral": (
        "A single flat-design shield icon. Classic heraldic shield shape, "
        "solid fill dark charcoal color, thin white inner border line. "
        "Pure white background. "
        "Absolutely NO text, NO letters, NO numbers, NO watermarks, NO labels, NO captions, NO shadows. "
        "Clean minimal icon. Centered in frame with padding."
    ),
    "magnifier_neutral": (
        "A single flat-design magnifying glass icon. Circle lens with a diagonal handle at lower-right. "
        "Solid dark charcoal color. Pure white background. "
        "Absolutely NO text, NO letters, NO numbers, NO watermarks, NO labels, NO captions, NO shadows. "
        "Clean minimal icon. Centered in frame with padding."
    ),
}

def generate_icon(name: str, prompt: str, size: int = 512):
    out_path = ICONS_DIR / f"{name}.png"
    if out_path.exists():
        print(f"  cached: {out_path.name}")
        return
    print(f"  generating: {name} ...")
    response = client.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            output_mime_type="image/png",
        ),
    )
    img_bytes = response.generated_images[0].image.image_bytes
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)

    # Make white/near-white pixels transparent
    data = img.getdata()
    new_data = []
    for r, g, b, a in data:
        if r > 220 and g > 220 and b > 220:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append((r, g, b, a))
    img.putdata(new_data)
    img.save(out_path, "PNG")
    print(f"  saved: {out_path}")


def tint_icon(name: str, hex_color: str, out_name: str, alpha: float = 1.0):
    """Load a neutral icon and tint it with a hex color."""
    src = ICONS_DIR / f"{name}.png"
    dst = ICONS_DIR / f"{out_name}.png"
    if dst.exists():
        print(f"  cached tint: {dst.name}")
        return
    img = Image.open(src).convert("RGBA")
    r_t = int(hex_color[1:3], 16)
    g_t = int(hex_color[3:5], 16)
    b_t = int(hex_color[5:7], 16)
    data = img.getdata()
    new_data = []
    for r, g, b, a in data:
        if a > 20:
            new_data.append((r_t, g_t, b_t, int(a * alpha)))
        else:
            new_data.append((0, 0, 0, 0))
    img.putdata(new_data)
    img.save(dst, "PNG")
    print(f"  tinted: {dst.name}")


FLASH_ICON_SPECS = {
    "building_crash": (
        "A single flat-style commercial office building icon, front-facing, pure white background. "
        "3 stories tall, slightly wider than tall, with a 3x2 grid of small square windows. "
        "Color: deep crimson red (#C0392B). Roofline has a cracked or jagged right edge implying damage. "
        "One top-floor window is visually darker/cracked. Bold minimal flat design. "
        "NO text, NO letters, NO numbers, NO watermarks, NO labels, NO shadows, NO gradients. "
        "Centered in frame with generous white padding on all sides."
    ),
    "building_stable": (
        "A single flat-style commercial office building icon, front-facing, pure white background. "
        "3 stories tall, slightly wider than tall, with a 3x2 grid of small square windows, all intact and bright. "
        "Color: rich forest green (#1E8449). Roofline is perfectly clean and straight. Conveys prosperity. "
        "Bold minimal flat design. NO text, NO letters, NO numbers, NO watermarks, NO labels, NO shadows, NO gradients. "
        "Centered in frame with generous white padding on all sides."
    ),
    "person_puppetmaster": (
        "A single flat-design icon of a shadowy puppet master figure, front-facing, pure white background. "
        "The figure wears a distinctive top hat. Both arms are extended outward and slightly raised, "
        "with thin vertical lines (puppet strings) hanging downward from each outstretched hand. "
        "Body is a simple rounded silhouette with a slightly hunched, menacing posture. "
        "Solid color: deep crimson red (#922B21). "
        "NO text, NO letters, NO numbers, NO watermarks, NO labels, NO shadows, NO gradients. "
        "Bold minimal icon style. Centered in frame with generous padding."
    ),
    "crash_chart_v2": (
        "A minimal data visualization chart icon, no text, no axis labels, pure graphic only. "
        "A thick bold crimson-red (#C0392B) price line starts high on the left and crashes steeply "
        "downward like a cliff edge, ending well below a horizontal dashed danger line near the bottom. "
        "The area BELOW the dashed danger floor line is filled with intense crimson-red at 30% opacity "
        "creating a striking danger zone. No gentle slope - the drop is sudden and dramatic. "
        "Pure white background. NO text, NO numbers, NO axis labels, NO tick marks, NO gridlines, "
        "NO title, NO watermarks, NO legends. Chart fills 80% of frame. Clean flat illustration style."
    ),
    "magnifier_forensic": (
        "A single flat-design magnifying glass icon, pure white background. "
        "Large circle lens with a thin circular border. Short thick handle at 45-degree angle from lower-right. "
        "Border ring and handle are bold orange (#B7600A). Interior of lens is white. "
        "NO text, NO letters, NO numbers, NO watermarks, NO labels, NO shadows, NO gradients. "
        "Clean minimal icon. Centered in frame with generous padding. Pure white background."
    ),
    "pipeline_traces": (
        "Three overlapping rectangular document icons arranged in a slight fan, slightly offset from each other. "
        "Each document has a tiny jagged sparkline chart drawn on its face representing market data. "
        "Flat minimal design. Bold orange-amber color (#B7600A). Pure white background. "
        "NO text, NO letters, NO numbers, NO watermarks, NO labels, NO shadows, NO gradients. "
        "Clean icon style, generous padding."
    ),
    "pipeline_filter": (
        "A single flat-design funnel/hopper icon. Wide opening at top, narrow spout at bottom. "
        "Three small dots enter the wide top, one dot exits the narrow bottom spout (filtering). "
        "Bold orange-amber color (#B7600A). Pure white background. "
        "NO text, NO letters, NO numbers, NO watermarks, NO labels, NO shadows, NO gradients. "
        "Minimal clean icon, generous padding."
    ),
    "brain_lora": (
        "A single flat-design icon of a stylized brain, classic bilobed rounded shape. "
        "Overlaid on the brain are two small bright rectangular patches like sticky notes, "
        "representing LoRA adapter modules attached to the model. "
        "Brain is forest green (#1E8449), adapter patches are white or light yellow. "
        "Pure white background. "
        "NO text, NO letters, NO numbers, NO watermarks, NO labels, NO shadows, NO gradients. "
        "Minimal flat icon, generous padding."
    ),
}


def generate_icon_flash(name: str, prompt: str, size: int = 512):
    """Generate an icon using gemini-2.5-flash-preview-04-17 with image output."""
    out_path = ICONS_DIR / f"{name}.png"
    if out_path.exists():
        print(f"  cached: {out_path.name}")
        return
    print(f"  generating (flash): {name} ...")
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    img_bytes = None
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            img_bytes = part.inline_data.data
            break
    if img_bytes is None:
        print(f"  WARNING: no image returned for {name}")
        return
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)

    # Make white/near-white pixels transparent
    data = img.getdata()
    new_data = []
    for r, g, b, a in data:
        if r > 220 and g > 220 and b > 220:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append((r, g, b, a))
    img.putdata(new_data)
    img.save(out_path, "PNG")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    print("=== Generating base icons ===")
    for name, prompt in ICON_SPECS.items():
        generate_icon(name, prompt)

    print("=== Tinting icons ===")
    tint_icon("building_neutral", "#C0392B", "building_red")       # crash firms
    tint_icon("building_neutral", "#1E8449", "building_green")     # stabilizing firms
    tint_icon("person_neutral",   "#922B21", "person_principal")   # sybil principal (dark red)
    tint_icon("person_neutral",   "#7D3C98", "person_sybil", alpha=0.65)  # sybil identity (purple, faded)
    tint_icon("person_neutral",   "#566573", "person_buyer")       # deceived buyer (grey)
    tint_icon("person_neutral",   "#1A5276", "person_guardian")    # guardian (dark blue)
    tint_icon("shield_neutral",   "#1A5276", "shield_blue")        # guardian shield
    tint_icon("magnifier_neutral","#B7600A", "magnifier_orange")   # forensic callout

    print("=== Generating flash icons ===")
    for name, prompt in FLASH_ICON_SPECS.items():
        generate_icon_flash(name, prompt)

    print("Done.")
