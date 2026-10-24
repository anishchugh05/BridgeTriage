---
inclusion: manual
---

# Frontend design rules (Emil Kowalski style)

Fast, subtle, polished. Motion should feel natural, never flashy.

## Motion

- Use the ease out curve cubic-bezier(0.23, 1, 0.32, 1) for all motion.
  Never use ease-in or the browser default ease.
- Every UI animation runs under 300ms. Hover and press feedback around 150ms.
- Animate only transform and opacity. Nothing else.
- Buttons and clickable cards scale to 0.97 on active (press).
- Never animate an element in from scale(0). Start at scale(0.95) opacity 0.
- Stagger list and card entrances by 40ms each, capped at 8 items.
- Headline number is large and counts up once on load in under 600ms.
- Respect prefers-reduced-motion: disable all animation when it is set.

## Type and numbers

- font-variant-numeric: tabular-nums on every number so values do not jitter.

## Surfaces and color

- Layered soft shadows instead of hard borders for cards.
- Subtle 1px inner highlight on dark surfaces.
- One accent color, red, for risk. Neutral grays for everything else.
- Plenty of whitespace.

## Feedback

- Use st.toast for feedback like "Picked 42 bridges", not loud banners.

## Implementation

- Inject all CSS through one st.markdown(unsafe_allow_html=True) block in
  app/style.py. Do not scatter CSS across the app.
- Keep code short and plain. No em dashes or en dashes anywhere.