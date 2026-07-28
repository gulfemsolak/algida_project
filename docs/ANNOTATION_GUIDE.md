# Annotation Guide

## Tool Recommendation

Use **Roboflow** (roboflow.com) — free tier supports up to 10,000 images with bounding box labelling, auto-exports in YOLO format, and includes version control for annotations.

Alternative: **LabelImg** (local, offline, open source).

---

## Class Definitions

Draw a **tight bounding box** around the full visible packaging of each product.

| Class | Description | Visual cue |
|-------|-------------|-----------|
| `magnum_classic` | Magnum Classic ice cream bar | Dark chocolate coating, white/gold Magnum logo |
| `magnum_almond` | Magnum Almond variant | Chocolate + almond pieces visible on packaging |
| `magnum_white` | Magnum White Chocolate | White/light packaging, "White" text |
| `cornetto_vanilla` | Cornetto vanilla cone | Blue/white packaging, cone shape |
| `cornetto_chocolate` | Cornetto chocolate cone | Brown/dark packaging |
| `cornetto_strawberry` | Cornetto strawberry cone | Pink/red packaging |
| `popsicle_fruit` | Fruit-flavoured ice lolly/popsicle | Bright colours (orange, red, green), stick visible |
| `popsicle_chocolate` | Chocolate-coated popsicle | Dark brown packaging or visible chocolate coating |
| `sandwich_ice_cream` | Ice cream sandwiched between biscuit layers | Rectangular, two dark biscuit layers visible |
| `cup_ice_cream` | Ice cream in a round cup/pot | Circular container, peel-off lid |
| `empty_slot` | Visible empty shelf position | No product — label the vacant space in the shelf grid |

---

## Bounding Box Guidelines

**DO:**
- Draw boxes **tight** around the full product/packaging
- Include the **entire packaging** — top, bottom, sides
- Annotate **every visible product**, even partially occluded ones
- Label `empty_slot` for each clearly empty grid position

**DON'T:**
- Include the shelf rail or background in the box
- Skip a product because it is tilted or partially hidden
- Draw one large box around a row of products
- Leave images unannotated (unlabelled images are excluded from training)

---

## Edge Cases

| Scenario | How to annotate |
|----------|----------------|
| Product tilted >45° | Still annotate; use the tightest upright bounding box |
| Product partially hidden by door reflection | Annotate visible portion; keep box tight |
| Two products touching | Draw separate boxes for each |
| Product fallen on its side | Annotate anyway; describe in notes |
| Damaged packaging, logo not visible | Annotate as best-guess class; mark as low-confidence |
| Stack of products | Annotate each individually if both are visible |

---

## Quality Standards

- Every product visible in the photo **must** be annotated
- Empty shelf slots must be annotated as `empty_slot`
- Minimum bounding box size: 10 × 10 pixels (smaller = probably noise, skip)
- Review at least 10% of annotations before export
- Export format: **YOLOv8** (normalized coordinates, `.txt` per image)

---

## Roboflow Workflow (Step-by-Step)

1. **Upload** images from `data/raw/`
2. **Annotate** each image using the classes above
3. **Generate** a dataset version (apply no preprocessing — we handle it ourselves)
4. **Export** as YOLOv8 format
5. **Download** the ZIP; copy `.txt` files to `data/annotations/`
6. Verify with `python -m src.data.validator`
