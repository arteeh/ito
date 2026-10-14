# GHOST-RITE

Owner-requested variant: designer-6's GHOST-LINE (Ghost in the Shell 1995), recoloured from
sea-green to cyan (after the owner's reference gist), with designer-2's VESPER ENGINE menu
background reworked to fit: the menu panel sits inside a binding circle (three orbits, twelve
divisions, a ring of inscribed code and hex), GHOST-LINE's read arcs on the outer orbit, and
designer-2's seal re-cut as a circuit sigil on both sides. Cyberpunk plus possession ritual.

Fonts, layout, title marks, safety colours and prompts are GHOST-LINE's unchanged
(Rajdhani, Ubuntu Mono, Michroma; see FONTS.md). Menu-only decoration; HUD untouched.
Static, no animation. About 150 draw calls on menus (mostly inscription glyphs).

    uv run python shot.py themes/ghost-rite --out shots/ghost-rite
    uv run python shot.py themes/ghost-rite --scale 1.25 --out shots/ghost-rite-125

## Thread sketch, take 2 (糸, ito)

- Ring: two quotes as lowercase hex byte pairs ("62 65 20 ..."), glyphs turned along the circle.
  Upper arc, read left to right: Pico della Mirandola, "be the free and proud shaper of your own
  being." Lower arc, also left to right: the Puppet Master (GitS 1995), "your effort to remain
  what you are is what limits you." `//` closes each at 3 and 9 o'clock (`QUOTES`). Spare room
  becomes even letter spacing; if a quote won't fit, the type shrinks, never squeezes.
- Seals: a quiet hanko, one thin ring with 糸 centred on its ink box, a thread through it.
  Kanji are drawn as glyphs, never hex.
- Marionette threads: the menu hangs from the outer orbit by three faint threads that pass
  behind the inscription.
- Running stitch (sashiko) inside the menu border.
- `link_thread(dl, a, b, state, u)`: taut (connected), slack (lagging), cut (糸が切れた, link lost);
  HUD title shows it taut. Sketch: `uv run python sketch_link.py themes/ghost-rite`.

## Motion (take 3)

`uv run python demo.py themes/ghost-rite` -> `shots/ghost-rite-demo/intro.mp4` (16 s, 60 fps) and
`frames.png` (stills from the first 2.2 s). Ito calls the theme's `intro(t)` hook at startup and when
it returns to the main menu after a session, never between menu pages. No reduce-motion switch.

- 0.0-1.1 s: the three orbits grow from a point at the screen centre, a beat apart.
- 0.45-1.05 s: the panel fades in (ImGui style alpha, set in `frame_begin`).
- 0.55-1.2 s: the twelve divisions tick round clockwise from twelve o'clock.
- 0.8-1.8 s: the hex quotes decode from the middle of each quote outwards, digits flicker first.
- 0.9-1.4 s: the two read arcs sweep open.
- 1.15-1.8 s: the marionette threads drop to the panel.
- 1.5-2.15 s: the seals fade in, left then right.
- Always: the quote ring turns once in 3 minutes, the read arcs drift the other way.
  The HUD and the safety prompts never move.

Still pure ImGui: draw lists, glyph quads from the font atlas and style alpha. No shaders, no textures.

## Fades (take 4)

- `//` joins sit in the middle of each gap between the two quotes (the quotes differ in
  length, so the gaps lean off the horizontal axis by a quarter of the difference).
- Every panel fades in and out in 0.1 s (`FADE = 0.1`): main menu, Settings, the HUD and the
  "Robot stopped"/E-STOP prompts. A screen change fades the old one out, then the new one in
  (0.2 s), so two panels never overlap; the circle stays put between menu pages and fades out
  with the menu when a session starts. The ~2 s intro still plays at startup and after a session.
- Theme API: `theme.show(name, kind, intro)`, `theme.showing()`, `theme.panel(name, visible)`,
  `theme.faded(alpha, leaving)`; a leaving panel is drawn read-only. Theme hook `screen(old, new, t)`.
