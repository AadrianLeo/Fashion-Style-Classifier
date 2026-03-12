# Fashion Style — Annotation Guidelines

This document defines labeling rules for the five fashion style classes.
Use these guidelines when annotating images in Label Studio or reviewing/correcting existing labels.

---

## Classes

### 1. `casual`
**Definition:** Everyday, relaxed, comfortable clothing worn outside the home but not for specific occasions.

**Visual cues:**
- T-shirts, polo shirts, plain sweatshirts
- Jeans, chinos, shorts
- Sneakers, loafers, slip-ons
- Relaxed or slim fit (not oversized)
- Muted, neutral, or basic colors

**Exclude if:** The look is clearly branded/hype (→ streetwear), athletic-focused (→ sporty), or decade-specific (→ vintage).

---

### 2. `formal`
**Definition:** Clothing appropriate for professional, business, or special-occasion settings.

**Visual cues:**
- Suits, blazers, sport coats
- Button-down dress shirts (tucked in)
- Dress trousers, pencil skirts
- Dress shoes, oxfords, heels
- Structured silhouette; tailored fit
- Neutral, dark, or classic colors

**Exclude if:** The blazer is worn casually over a hoodie without dress trousers (→ casual).

---

### 3. `sporty`
**Definition:** Clothing primarily designed for athletic activity or that strongly evokes sport/fitness.

**Visual cues:**
- Compression leggings, running shorts, gym shorts
- Sports jerseys, track jackets, windbreakers
- Athletic sneakers (e.g., running shoes, court shoes)
- Technical fabrics (dry-fit, spandex, mesh)
- Brand logos from sport brands (Nike, Adidas, Under Armour) in athletic contexts

**Exclude if:** The tracksuit is fashion-branded and clearly worn as streetwear (→ streetwear).

---

### 4. `streetwear`
**Definition:** Urban, culture-driven style characterized by hype brands, oversized fits, and subcultural references.

**Visual cues:**
- Oversized hoodies, graphic tees
- Cargo pants, joggers, baggy jeans
- Chunky sneakers, high-tops (Jordan, Yeezy, New Balance)
- Visible brand logos (Supreme, Off-White, Palace, Stüssy, etc.)
- Layering; unconventional color combinations or prints
- Caps, beanies, bucket hats

**Exclude if:** The look is clearly athletic with technical fabrics only (→ sporty).

---

### 5. `vintage`
**Definition:** Clothing that evokes a specific past decade (pre-2000s) in cut, print, or aesthetic.

**Visual cues:**
- High-waisted trousers or skirts (70s/80s silhouette)
- Retro patterns (plaid, floral, paisley in classic cuts)
- Band tees, concert tees from older acts
- Corduroy, denim jacket with patches
- Thrift-store aesthetic; faded or worn colors
- Silhouettes typical of 1950s–1990s fashion

**Exclude if:** The retro print is on a clearly modern/slim-fit garment that reads as casual.

---

## Ambiguity Rules

1. **When in doubt, pick the most prominent/defining element** of the outfit (shoes + pants + top together, not just one item).
2. **Casual vs. streetwear** — key differentiator is brand presence and deliberate subcultural styling. A plain white tee + jeans = casual; a Supreme hoodie + Jordans = streetwear.
3. **Sporty vs. streetwear** — key differentiator is functional athletic intent. Running shoes + dry-fit shirt = sporty; the same shoes with a graphic tee and cargo pants = streetwear.
4. **Vintage vs. casual** — key differentiator is decade-specificity. If you can identify a clear decade, label it vintage.
5. **Skip** images that show only accessories (bags, hats) with no outfit visible.

---

## Inter-Annotator Agreement Target
- Cohen's κ ≥ 0.70 on 10% sample review
- Images with κ contribution < 0.5 (annotators disagree) are flagged for discussion

---

## Export Format
Export from Label Studio as CSV with columns:
```
annotation_id, annotator, choice, created_at, id, image, lead_time, updated_at
```
This matches `Dataset/labels_file.csv`.
