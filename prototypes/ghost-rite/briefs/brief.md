# Design an original ImGui theme + font for Ito — you are designer __ID__

Ito is the owner's telepresence app: a pilot drives a small robot (Microduck) and sees a live
gaussian-splat reconstruction of its surroundings, on desktop or in a VR headset. Its UI is
Dear ImGui (Python, imgui_bundle 1.92.900). Today it uses ImGui's stock dark style and the
built-in bitmap font. The owner wants something unique and stylish instead.

Owner's keywords: **futuristic, cyberpunk, exotic, esoteric.**
Be ambitious. Ito does not have to be corporate-safe. The owner likes this theme as an example
(though he calls it "not very ambitious"):
https://gist.github.com/enemymouse/c8aa24e247a1d7b9fc33d45091cbb8f0 — do NOT copy it; make
something original. Three designers work on this in parallel, each alone; your lean (a nudge,
not a rule — go wherever your idea is strongest): __LEAN__

## The lab (ready, tested)

`~/Projects/ito-theme-lab` (this directory). Read README.md, `lab/theme_api.py` (the theme
contract: fonts per role, full ImGui style, semantic colours, draw-list hooks under/over each
window and behind/in front of everything) and `lab/screens.py` (Ito's real screens, widget for
widget). Baseline: `uv run python shot.py themes/ito-default` -> `shots/ito-default/sheet.png`.
Render is headless (EGL); you can look at the PNGs you produce, so do: iterate by eye.

## Do

1. Find fonts online: unique, cool-looking, but **free and redistributable** vector fonts
   (TTF/OTF; SIL OFL, Apache 2.0, MIT, Ubuntu Font Licence, CC0/public domain — NOT "free for
   personal use", not unclear licences). Look beyond the obvious (Orbitron, Share Tech Mono are
   everywhere): Google Fonts, Fontsource, GitHub, Velvetyne, Collletttivo, Open Foundry, etc.
   Download into `themes/__ID__/fonts/` with each font's licence file. A striking display font
   for titles/headings plus a legible body (and maybe numeric/mono) font is a good pattern.
2. Build `themes/__ID__/theme.py` per `lab/theme_api.py`: fonts and sizes, every relevant
   ImGui style var and colour, the semantic COLORS, and draw hooks if they serve the idea
   (frames, corner brackets, scanlines, glyph ornaments, glows, gradients ... your call).
3. Render: `uv run python shot.py themes/__ID__` and
   `uv run python shot.py themes/__ID__ --scale 1.25 --out shots/__ID__-125`. Look at every
   PNG, refine, repeat until you are proud of it.
4. Write `themes/__ID__/FONTS.md` (each font: name, designer, source URL, licence) and
   `themes/__ID__/NOTES.md` (the concept in a few lines, palette, what makes it original, any
   caveats).

## Must hold

- Ito's safety text must stay unmistakable: E-STOP LATCHED, Robot stopped, errors, warnings.
  Body text and numbers must stay readable at HUD sizes and in a VR headset (no hairline-thin
  body strokes, no 1 px detail that shimmers in VR, enough contrast over a bright scene).
- Only the lab's hooks: the theme must work in Ito by dropping in theme.py + fonts, so do
  not edit `lab/`, `shot.py`, `themes/ito-default` or other designers' directories. If you
  need a hook the API lacks, describe it in NOTES.md instead.
- Hooks run every frame (also at 90 Hz in the headset): keep them cheap.
- Work only in `themes/__ID__/` and `shots/__ID__*`. Do not touch ~/Projects/ito or any
  other project, no git push, no GitHub issues/PRs. This box also runs a Minecraft server:
  nothing heavy.

## Final message

The theme's name and concept in 2-3 lines, the fonts and their licences, and the path to
`shots/__ID__/sheet.png`.
