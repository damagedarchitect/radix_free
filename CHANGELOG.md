# Changelog

All notable changes to Radix are documented here.

---

## [1.0.0] | 2026-06-30

Initial public release of **Radix** (free edition).

### What's included
- 39 snap positions (faces, corners, edge midpoints, centers, geometry extremes)
- Surface Snap | click any mesh surface; Shift for face centre
- Vertex Snap |click or drag to nearest vertex
- Grid Snap | round origin to world-grid increment
- Snap History | 10-slot ring buffer, auto-recorded on every snap; Prev/Next navigation with grayed-out buttons at boundaries
- Viewport preview | translucent highlight quad + direction arrow, toggled from panel; clears immediately when toggled off
- BBox Handle mode | 26 clickable handles in the viewport with hover-enlargement (1.55×)
- Origin → World Zero |  one button, no modal
- Edit Mode Snap | snap to selection centre in Edit Mode
- Quick snaps | Alt+Click (surface), Alt+Shift+Click (vertex), Alt+Ctrl+Click (face centre)
- Radix pie menu | Alt+Q
- Blender 4.5+ / 5.x compatible (extension format with `blender_manifest.toml`)

---