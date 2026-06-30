# Radix | A 39 Precision Origin Placement for Blender

**Radix** snaps object origins to exactly the right place with one click to any face, corner, edge midpoint, surface point, or vertex on any mesh.

Built for architects, product designers, and anyone who needs clean pivot points without fighting Blender's default origin tools.

---

## Features

### 39 Snap Positions
Snap to any face, corner, edge midpoint, geometry extreme, or center of the active mesh which  is visible as a live highlight quad and direction arrow in the viewport.

### Surface & Vertex Snap
Click any mesh surface to snap the origin to that exact point. Hold **Shift** for the face centre instead. Or click any vertex directly.

### Grid Snap
Round the origin to the nearest world-grid increment that keeps your pivots on-grid without manual coordinate entry.

### Snap History
Every origin move is recorded automatically. Step back and forward through the last 10 positions with **Prev / Next** without undoing your geometry changes.

### Viewport Preview
A live translucent highlight shows which face the origin sits on, with a direction arrow indicating the local front axis. Toggle on and off from the panel.

### BBox Handle Mode
Click any of the 26 bounding-box handles directly in the viewport to snap the origin to corners, face centres, edge midpoints, and the volume centre.

### Origin → World Zero
One button. Moves the origin to (0, 0, 0) with no modal, no options.

### Edit Mode Snap
Snap the origin to the centre of the current selection while in Edit Mode.

### Handle Hover
Handles enlarge when the mouse passes over them so you always know which point will be committed before clicking.

---

## Installation

1. Download **radix-1.0.0.zip** from [Releases](../../releases/latest)
2. In Blender: **Edit → Preferences → Add-ons → Install from Disk**
3. Select the zip then Radix appears in the **N-Panel → Radix Pro** tab

Requires **Blender 4.5 or later** (including Blender 5.x).

---

## Usage

Open the **N-Panel** (press `N` in the 3D Viewport) and click the **Radix Pro** tab.

- **Preview Controls** | toggle the highlight quad and direction arrow
- **Set Origin To** | the full snap grid; click any button to move the origin
- **Surface / Vertex / Grid / Cursor** | interactive snaps; click then click a surface
- **Snap History** | Prev / Next step through previous positions
- **BBox Handles** | enable viewport handle mode and click a handle directly

### Keyboard shortcuts (default)
| Shortcut | Action |
|---|---|
| `Alt + Click` | Quick surface snap (no panel needed) |
| `Alt + Shift + Click` | Quick vertex snap |
| `Alt + Ctrl + Click` | Quick face-centre snap |
| `Alt + Q` | Radix pie menu |

---

## Radix Pro

The free version covers the core workflow. **[Radix Pro](https://discord.com/users/damagedarchitect)** adds:

| Feature | Free | Pro |
|---|:---:|:---:|
| 39 snap positions | ✓ | ✓ |
| Surface / Vertex / Grid snap | ✓ | ✓ |
| Snap History (10 slots) | ✓ | ✓ |
| Viewport preview | ✓ | ✓ |
| BBox handles + hover | ✓ | ✓ |
| Origin → World Zero | ✓ | ✓ |
| Surface modifier keys (Shift / Ctrl / Alt) | — | ✓ |
| Axis-locked surface snap (X/Y/Z) | — | ✓ |
| Numerical offset input while snapping | — | ✓ |
| Placement Tools (offsets, axis locks, live offset) | — | ✓ |
| Normal Offset | — | ✓ |
| Copy / Paste origin | — | ✓ |
| Multi-Object preview | — | ✓ |
| Object Snap Tools | — | ✓ |
| Snap History HUD overlay | — | ✓ |
| Viewport Mode Indicator | — | ✓ |
| Radix Place suite | — | ✓ |
| → Pivot Library (8 named slots per object) | — | ✓ |
| → Collision Preview (live bbox overlap) | — | ✓ |
| → Surface Alignment (snap + rotate to normal) | — | ✓ |
| → Smart Contact Detection | — | ✓ |
| → Snap Layers (named setting presets) | — | ✓ |
| → Chain / Distribute origins | — | ✓ |
| → Batch Contact Detection | — | ✓ |

→ **[Request Radix Pro on Discord](https://discord.com/users/damagedarchitect)**

---

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE).

---

## Contributing

Bug reports and feature requests are welcome via [Issues](../../issues).

Pull requests are reviewed but may not always be merged | Radix Pro is a commercial product and significant feature additions are kept there. Small fixes, documentation improvements, and compatibility patches are the most likely to be accepted.

---

*Made by [damagedarchitect] | Abu Dhabi, UAE*