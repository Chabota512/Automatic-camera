"""Create the Windows icon without requiring a graphics package beyond Pillow."""

from pathlib import Path

from PIL import Image, ImageDraw


def build_icon(size: int = 256) -> Image.Image:
    scale = 4
    canvas_size = size * scale
    image = Image.new("RGBA", (canvas_size, canvas_size), (23, 35, 51, 255))
    draw = ImageDraw.Draw(image)

    def box(values):
        return tuple(int(value * scale) for value in values)

    draw.rounded_rectangle(
        box((0, 0, size, size)),
        radius=48 * scale,
        fill="#172333",
    )
    draw.rounded_rectangle(
        box((53, 65, 203, 194)),
        radius=23 * scale,
        fill="#2dd4bf",
    )
    draw.rectangle(box((104, 46, 152, 78)), fill="#2dd4bf")
    draw.ellipse(box((79, 78, 177, 176)), fill="#0f172a")
    draw.ellipse(box((97, 96, 159, 158)), fill="#38bdf8")
    draw.ellipse(box((108, 107, 148, 147)), fill="#172333")

    stroke = 7 * scale
    for start, end in (
        ((128, 81), (128, 99)),
        ((128, 155), (128, 173)),
        ((82, 127), (100, 127)),
        ((156, 127), (174, 127)),
    ):
        draw.line((*box(start), *box(end)), fill="#f8fafc", width=stroke)
    draw.ellipse(box((170, 84, 184, 98)), fill="#f8fafc")
    return image.resize((size, size), Image.Resampling.LANCZOS)


if __name__ == "__main__":
    output = Path(__file__).with_name("automatic-camera.ico")
    icon = build_icon()
    icon.save(
        output,
        format="ICO",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
    )
    print(f"Wrote {output}")