# Ito theme lab

A standalone slice of Ito's ImGui screens (copied from Ito main ff9993b, fake data, no robot)
for designing an ImGui theme + fonts and screenshotting it headless.

    uv run python shot.py themes/ito-default            # today's look -> shots/ito-default/
    uv run python shot.py themes/<name>                 # a theme -> shots/<name>/
    uv run python shot.py themes/<name> --scale 1.25 --out shots/<name>-125

Each run writes 1-menu, 2-settings, 3-hud (Robot stopped prompt, pilot trying to drive),
4-hud-alert (E-STOP latched, out-of-memory warnings) and sheet.png (all four).

- `lab/screens.py`: Ito's connect menu, Settings page, pilot HUD and stop prompt, widget for widget.
- `demo.py`: renders a theme's motion (intro, menu pages, a session) to `shots/<name>-demo/intro.mp4`.
- `lab/theme_api.py`: what a theme may define (fonts, style, semantic colours, draw hooks).
- `themes/ito-default/`: Ito today (ImGui dark, ProggyClean 13 px).
