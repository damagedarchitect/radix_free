bl_info = {
    "name": "Radix Free",
    "blender": (4, 5, 0),  # Compatible through Blender 5.1+
    "category": "Object",
    "description": (
        "Precision origin placement with 39 snap positions across faces, vertices, edges, "
        "and centers — accessible via dropdown menus and clickable viewport handles."
    ),
    "author": "DaMagedArchitect",
    "version": (1, 0, 0),
    "doc_url": "",
    "tracker_url": "",
}


import bpy
import csv
import os
import mathutils
from mathutils import Vector, Matrix
import gpu
from gpu_extras.batch import batch_for_shader
import math
from bpy_extras.view3d_utils import (
    region_2d_to_vector_3d,
    region_2d_to_origin_3d,
    location_3d_to_region_2d,
)

# ---------------- Constants ----------------
EPS = 1e-9


def deselect_all_objects(context):
    """Deselect every object in the view layer without bpy.ops.object.select_all.

    bpy.ops.object.select_all polls for an active 3D Viewport context and raises
    RuntimeError when invoked from an N-panel button (or any non-viewport context).
    Iterating view_layer.objects and calling select_set(False) has no poll
    requirement and works safely from any operator execute() or callback context.
    """
    for o in context.view_layer.objects:
        try:
            o.select_set(False)
        except Exception:
            pass
PREVIEW_ARROW_NAME = "RadixPro_Preview_Arrow"
PREVIEW_COLLECTION = "RadixPro_Preview_Col"

# Module-level handler reference
_draw_handler_3d = None
_draw_handler_2d = None   # POST_PIXEL for shaped bbox handles
_draw_handler_outliner = None  # POST_PIXEL for per-object front-axis dot in outliner

# Re-entry guard for depsgraph handler
_depsgraph_updating = False

# Track active object name to detect selection changes for front axis reset
_last_active_object_name = None

# ── New in v1.2.0 ────────────────────────────────────────────────────────────

# Origin history: {obj_name: [Vector(world_pos), ...]}  max 5 per object
_origin_history: dict = {}
ORIGIN_HISTORY_MAX = 5

# Copy/Paste clipboard  {'position': Vector, 'front_axis': str, 'snap_source': str}
_origin_clipboard: dict = {}

# Proportional-origin clipboard: normalized position (0..1 per axis) of an
# object's origin within its own local bounding box. Captured from a reference
# object, applied to others so each gets the SAME relative origin placement
# regardless of its size. {'t': Vector} or empty.
_origin_proportion_clipboard: dict = {}

# Module-level cache dictionary for GPU overlay
# quad_local: quad verts in object local space (always)
# obj_name: which object owns the cache
# orientation/front_axis: needed to recompute GLOBAL quads at draw time
_preview_cache = {
    'quad_world': None,
    'quad_local': None,
    'obj_name':   None,
    'orientation': None,
    'front_axis':  None,
    'bbox_local':  None,
}

# Per-object preview storage (when Keep Persistent is enabled)
# Structure: {
#   'custom_color': (r,g,b) or None,     # User's RGB picker override
#   'collection_color': (r,g,b) or None, # From Blender's collection tag
#   'fallback_color': (r,g,b),           # Random default
#   'muted': False,
#   'arrow': object_reference,
#   'quad': [v1, v2, v3, v4]
# }
_persistent_previews = {}

# Timer handle for auto-show
_auto_show_timer = None

# Last confirmed snap target per object in LOCAL space.
# Keyed by object name. Converted world→local at recording time,
# local→world at placement time so the arrow moves with the object.
_last_snap_target = {}

# Last surface normal at snap point per object — used by Normal Offset
# {obj_name: Vector(world-space normal)}
_last_snap_normals: dict = {}

# Hovered handle world position — written by RADIX_OT_click_bbox_handle.modal()
# on every MOUSEMOVE, read by draw_handle_shapes_2d() to scale the hovered shape.
# Single-element list so it's mutable from inside class methods without `global`.
_handle_hover_world_pos: list = [None]  # None | Vector

# ── Radix Place — Phase 1: Snap History ───────────────────────────────────────
_SNAP_HISTORY: list       = []
_SNAP_HISTORY_MAX: int    = 10
_SNAP_HISTORY_IDX: list   = [0]    # mutable container; avoids `global` in methods
_SNAP_RECORDING:  list    = [True]  # set False during history replay / batch ops
                                    # to prevent re-recording while restoring

# ── Radix Place — Phase 1: Pivot Library ──────────────────────────────────────
_PIVOT_SLOTS: int         = 8
# Key stored ON the object — slot number only, no obj.name in key.
# This means pivots survive object renames.
PIVOT_PROP_KEY: str       = 'radixpro_pivot_{}'   # format(slot)

# ── Radix Place — Phase 1: Collision Preview ──────────────────────────────────
_collision_handler: list  = [None]   # [draw_handler | None]
_collision_active:  list  = [False]  # [bool]

# ── Snap History HUD (F3) ─────────────────────────────────────────────────────
# A small POST_PIXEL overlay shown for ~2 s after Prev/Next navigation.
_history_hud_handler: list = [None]   # [draw_handler | None]
_history_hud_text:    list = [None]   # [str | None]
_history_hud_time:    list = [0.0]    # [float] — time.monotonic() when last set

# ── Viewport Mode Indicator (F6) ──────────────────────────────────────────────
_mode_ind_handler: list = [None]   # [draw_handler | None]

# Surface snap mode state
_surface_snap_active = False
_surface_snap_preview_point = None

# GLOBAL COLOR DEFINITIONS (use everywhere)
COLOR_PRESETS = {
    'RED': {
        'rgb': (0.9, 0.2, 0.2),
        'collection_icon': 'COLLECTION_COLOR_01',
        'colorset_icon': 'COLORSET_01_VEC',
        'name': 'Red'
    },
    'ORANGE': {
        'rgb': (0.95, 0.5, 0.2),
        'collection_icon': 'COLLECTION_COLOR_02',
        'colorset_icon': 'COLORSET_02_VEC',
        'name': 'Orange'
    },
    'YELLOW': {
        'rgb': (0.95, 0.85, 0.2),
        'collection_icon': 'COLLECTION_COLOR_03',
        'colorset_icon': 'COLORSET_03_VEC',
        'name': 'Yellow'
    },
    'GREEN': {
        'rgb': (0.3, 0.8, 0.3),
        'collection_icon': 'COLLECTION_COLOR_04',
        'colorset_icon': 'COLORSET_04_VEC',
        'name': 'Green'
    },
    'BLUE': {
        'rgb': (0.2, 0.4, 0.9),
        'collection_icon': 'COLLECTION_COLOR_05',
        'colorset_icon': 'COLORSET_05_VEC',
        'name': 'Blue'
    },
    'PURPLE': {
        'rgb': (0.7, 0.3, 0.9),
        'collection_icon': 'COLLECTION_COLOR_06',
        'colorset_icon': 'COLORSET_06_VEC',
        'name': 'Purple'
    },
    'PINK': {
        'rgb': (0.9, 0.2, 0.7),
        'collection_icon': 'COLLECTION_COLOR_07',
        'colorset_icon': 'COLORSET_07_VEC',
        'name': 'Pink'
    },
    'BROWN': {
        'rgb': (0.6, 0.4, 0.2),
        'collection_icon': 'COLLECTION_COLOR_08',
        'colorset_icon': 'COLORSET_08_VEC',
        'name': 'Brown'
    },
}

# Blender's native collection color tag mapping
BLENDER_COLLECTION_COLORS = {
    'COLOR_01': (0.9, 0.2, 0.2),    # Red
    'COLOR_02': (0.95, 0.5, 0.2),   # Orange
    'COLOR_03': (0.95, 0.85, 0.2),  # Yellow
    'COLOR_04': (0.3, 0.8, 0.3),    # Green
    'COLOR_05': (0.2, 0.4, 0.9),    # Blue/Cyan
    'COLOR_06': (0.7, 0.3, 0.9),    # Purple
    'COLOR_07': (0.9, 0.2, 0.7),    # Pink/Magenta
    'COLOR_08': (0.6, 0.4, 0.2),    # Brown
}


# ── History helpers ───────────────────────────────────────────────────────────

def _record_history(obj):
    """Push current world origin onto obj's history stack (max 5)."""
    global _origin_history
    pos = obj.matrix_world.translation.copy()
    stack = _origin_history.setdefault(obj.name, [])
    if stack and (stack[-1] - pos).length < 1e-6:
        return
    stack.append(pos)
    if len(stack) > ORIGIN_HISTORY_MAX:
        stack.pop(0)


def _apply_origin(obj, world_pos, context):
    """Move obj origin to world_pos via cursor. Saves/restores selection and mode.

    Mode switching and VIEW_3D context injection are handled by _safe_origin_set,
    so _apply_origin only needs to manage selection and cursor state.
    """
    saved_sel    = list(context.selected_objects)
    saved_active = context.view_layer.objects.active
    saved_cursor = context.scene.cursor.location.copy()

    try:
        deselect_all_objects(context)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        context.scene.cursor.location = world_pos
        _safe_origin_set(context, type='ORIGIN_CURSOR', center='MEDIAN')
        context.view_layer.update()
        # Persist the offset that produced this snap so the user can recall it
        try:
            scene = context.scene
            ox = getattr(scene, 'origin_offset_x', 0.0)
            oy = getattr(scene, 'origin_offset_y', 0.0)
            oz = getattr(scene, 'origin_offset_z', 0.0)
            if abs(ox) > 1e-9 or abs(oy) > 1e-9 or abs(oz) > 1e-9:
                obj['radixpro_last_offset_x'] = ox
                obj['radixpro_last_offset_y'] = oy
                obj['radixpro_last_offset_z'] = oz
        except Exception:
            pass
        # Auto-record every successful snap into Radix Place Snap History
        try:
            push_snap_history(obj.name, Vector(world_pos))
        except Exception:
            pass
    finally:
        context.scene.cursor.location = saved_cursor
        deselect_all_objects(context)
        for o in saved_sel:
            if o and o.name in context.view_layer.objects:
                o.select_set(True)
        if saved_active and saved_active.name in context.view_layer.objects:
            context.view_layer.objects.active = saved_active


_ORIGIN_SET_MODE_MAP = {
    'EDIT_MESH':    'EDIT',
    'EDIT_CURVE':   'EDIT',
    'EDIT_ARMATURE':'EDIT',
    'POSE':         'POSE',
    'SCULPT':       'SCULPT',
    'PAINT_WEIGHT': 'WEIGHT_PAINT',
    'PAINT_VERTEX': 'VERTEX_PAINT',
    'PAINT_TEXTURE':'TEXTURE_PAINT',
    'PARTICLE':     'PARTICLE_EDIT',
}


def _find_view3d_ctx(context):
    """Return (window, area, region) for the first VIEW_3D WINDOW region found.

    Operators like mode_set and origin_set poll for a VIEW_3D context. When
    called from an N-panel button, the context area is the N-panel itself —
    not the viewport — so those polls fail. By finding a real VIEW_3D area we
    can pass it to context.temp_override() and satisfy the poll from any calling
    context, including N-panel buttons, timers, and depsgraph handlers.

    Returns (None, None, None) if no 3D viewport is open.
    """
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for region in area.regions:
                if region.type == 'WINDOW':
                    return window, area, region
    return None, None, None


def _safe_origin_set(context, type='ORIGIN_CURSOR', center='MEDIAN'):
    """Call bpy.ops.object.origin_set with a guaranteed VIEW_3D context.

    origin_set — and mode_set — poll for an active VIEW_3D area. Invoking them
    from an N-panel button, timer, or callback gives the wrong area type, making
    the poll fail with RuntimeError even when we are in the right mode.

    Fix: find an actual VIEW_3D area via context.window_manager and inject it
    through context.temp_override() (Blender 3.2+, stable in 4.x/5.x). This
    gives both mode_set and origin_set a valid context without relying on
    whatever area triggered the operator call.
    """
    saved_mode = context.mode
    win, area, region = _find_view3d_ctx(context)

    def _run(fn, *args, **kwargs):
        if win and area and region:
            with context.temp_override(window=win, area=area, region=region):
                return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)   # no VIEW_3D open — last-resort attempt

    if saved_mode != 'OBJECT':
        try:
            _run(bpy.ops.object.mode_set, mode='OBJECT')
        except Exception as e:
            print(f"[Radix] _safe_origin_set: mode_set to OBJECT failed: {e}")
            return  # can't place origin without Object Mode — abort cleanly

    try:
        _run(bpy.ops.object.origin_set, type=type, center=center)
    finally:
        if saved_mode != 'OBJECT' and context.mode == 'OBJECT':
            restore = _ORIGIN_SET_MODE_MAP.get(saved_mode, 'OBJECT')
            try:
                _run(bpy.ops.object.mode_set, mode=restore)
            except Exception as e:
                print(f"[Radix] _safe_origin_set: mode restore to {restore} failed: {e}")


def _set_origin_world_fast(obj, world_pos, context=None):
    """Move obj's origin to world_pos WITHOUT bpy.ops and WITHOUT per-tick undo.

    Geometry stays visually fixed; only the pivot moves. Used by the live offset
    drag path. Because no operator is invoked, a continuous slider drag collapses
    into Blender's single natural undo step (which snapshots the moved geometry
    too) instead of flooding the undo stack with one origin_set per tick — and we
    avoid calling an operator from a property update callback, which is unsafe.

    The math: L = matrix_world⁻¹ @ world_pos is the target origin expressed in the
    object's current local frame. Shifting mesh data by -L places that point at
    local (0,0,0); re-stamping the world translation to world_pos (keeping the
    existing rotation/scale) leaves every vertex in the same world location.
    Verified for full TRS matrices and parented objects (matrix_world setter
    resolves the parent inverse).

    Falls back to the operator-based _apply_origin for multi-user mesh data (which
    can't be edited per-instance without moving the other instances) and for any
    non-mesh object.
    """
    data = getattr(obj, "data", None)
    if (obj.type != 'MESH' or data is None
            or not hasattr(data, "transform") or data.users > 1):
        if context is not None:
            _apply_origin(obj, world_pos, context)
        return

    mw = obj.matrix_world
    world_pos = Vector(world_pos)
    local_origin = mw.inverted() @ world_pos          # target origin in current local space
    data.transform(mathutils.Matrix.Translation(-local_origin))

    m = mw.copy()
    m.translation = world_pos                         # keep R/S, re-stamp translation
    obj.matrix_world = m
    data.update()


def _apply_lock_and_offset(obj, target, scene):
    """Apply axis-lock + numeric offset to a world-space snap target.
    In Direct mode the offset fields are absolute coords, not a delta — skip addition.
    In Object-space offset mode, offset is applied along the object's local axes."""
    cur = obj.matrix_world.translation.copy()
    is_direct = getattr(scene, 'origin_offset_mode', 'OFFSET') == 'DIRECT'
    if is_direct:
        return Vector((
            cur.x if scene.origin_lock_x else target.x,
            cur.y if scene.origin_lock_y else target.y,
            cur.z if scene.origin_lock_z else target.z,
        ))

    # Build offset vector
    raw_offset = Vector((scene.origin_offset_x, scene.origin_offset_y, scene.origin_offset_z))

    # Feature 3: object-space offset — rotate offset vector by object's world rotation
    offset_space = getattr(scene, 'origin_offset_space', 'WORLD')
    if offset_space == 'LOCAL' and raw_offset.length > 0:
        rot_mat = obj.matrix_world.to_3x3().normalized()
        world_offset = rot_mat @ raw_offset
    else:
        world_offset = raw_offset

    return Vector((
        (cur.x if scene.origin_lock_x else target.x) + world_offset.x,
        (cur.y if scene.origin_lock_y else target.y) + world_offset.y,
        (cur.z if scene.origin_lock_z else target.z) + world_offset.z,
    ))



def _live_offset_update(self, context):
    """Called whenever an offset field changes — moves origin live.

    Two modes controlled by origin_offset_mode:
      OFFSET  — fields are a delta added on top of the last snap base position.
      DIRECT  — fields ARE the world position. Drag to place origin anywhere.
    """
    scene = context.scene
    obj   = context.active_object
    if not obj or obj.type != 'MESH':
        return
    if not getattr(scene, 'origin_live_offset', False):
        return

    mode = getattr(scene, 'origin_offset_mode', 'OFFSET')

    if mode == 'DIRECT':
        # Fields = absolute world X Y Z.  Lock axes freeze that axis to current position.
        cur = obj.matrix_world.translation.copy()
        target = Vector((
            cur.x if scene.origin_lock_x else scene.origin_offset_x,
            cur.y if scene.origin_lock_y else scene.origin_offset_y,
            cur.z if scene.origin_lock_z else scene.origin_offset_z,
        ))
    else:
        # OFFSET mode — fields are delta from last snap base
        base_key = '_radixpro_offset_base'
        base = scene.get(base_key)
        if base is None:
            base = list(obj.matrix_world.translation)
            scene[base_key] = base
        base_vec = Vector(base)
        target = Vector((
            base_vec.x + scene.origin_offset_x,
            base_vec.y + scene.origin_offset_y,
            base_vec.z + scene.origin_offset_z,
        ))

    # Live drag: use the no-op setter so the whole drag is ONE undo step and no
    # operator runs inside this property callback.
    _set_origin_world_fast(obj, target, context)
    _last_snap_target[obj.name] = Vector((0, 0, 0))

    if scene.show_origin_preview:
        try:
            create_or_update_preview(obj)
        except Exception:
            pass  # intentional: try:


def _live_offset_update_x(self, context):
    if not context.scene.get('_radixpro_suppress_offset_update'):
        _live_offset_update(self, context)
def _live_offset_update_y(self, context):
    if not context.scene.get('_radixpro_suppress_offset_update'):
        _live_offset_update(self, context)
def _live_offset_update_z(self, context):
    if not context.scene.get('_radixpro_suppress_offset_update'):
        _live_offset_update(self, context)


def _offset_mode_update(self, context):
    """When switching to DIRECT mode, load current origin into the fields.
    When switching to OFFSET mode, reset fields to zero."""
    scene = context.scene
    obj   = context.active_object
    if not obj or obj.type != 'MESH':
        return
    if scene.origin_offset_mode == 'DIRECT':
        # Populate fields with current world origin — ready to nudge
        pos = obj.matrix_world.translation
        scene['_radixpro_offset_base'] = None  # clear offset base
        # Suppress the update callback while we set values
        scene['_radixpro_suppress_offset_update'] = True
        scene.origin_offset_x = round(pos.x, 6)
        scene.origin_offset_y = round(pos.y, 6)
        scene.origin_offset_z = round(pos.z, 6)
        scene['_radixpro_suppress_offset_update'] = False
    else:
        # Back to OFFSET — reset fields to zero, stash current position as base
        pos = obj.matrix_world.translation
        scene['_radixpro_offset_base'] = list(pos)
        scene['_radixpro_suppress_offset_update'] = True
        scene.origin_offset_x = 0.0
        scene.origin_offset_y = 0.0
        scene.origin_offset_z = 0.0
        scene['_radixpro_suppress_offset_update'] = False

def _deferred_preview_build():
    """One-shot timer: build preview after the property update has fully committed."""
    try:
        scene = bpy.context.scene
        obj   = bpy.context.active_object
        if (obj and obj.type in GEOMETRY_TYPES
                and getattr(scene, 'show_origin_preview', False)):
            create_or_update_preview(obj)
    except Exception as e:
        print(f"[Radix] deferred preview build failed: {e}")
    return None


def _deferred_preview_reset():
    """One-shot timer: clean up arrows and state after preview is toggled off.
    Must run from a timer — removing objects from bpy.data inside a property
    update callback causes a hard crash."""
    global _preview_cache, _persistent_previews
    try:
        _preview_cache.clear()
        _persistent_previews.clear()
        remove_preview_objects()
        _tag_view3d_redraw()
    except Exception as e:
        print(f"[Radix] deferred preview reset failed: {e}")
    return None


def auto_enable_mo_mode_on_multi_select(scene):
    """No-op: Multi-Object preview is now a derived computed property (mom_preview_active).
    Kept to avoid AttributeError from any call sites that weren't updated."""
    return False


# ---------------- Utilities (evaluated mesh + bbox) ----------------
def get_evaluated_mesh(obj):
    """Return (obj_eval, mesh) for evaluated object (modifiers applied).
    Forces a view_layer update so the depsgraph is never stale after origin_set calls.
    Raises RuntimeError if mesh cannot be evaluated (caller should treat as no geometry).
    """
    ctx = bpy.context
    # Ensure depsgraph is up to date — critical after origin_set dirtied the graph
    try:
        ctx.view_layer.update()
    except Exception:
        pass  # intentional: try:
    depsgraph = ctx.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    if mesh is None:
        raise RuntimeError(f"to_mesh() returned None for '{obj.name}'")
    if len(mesh.vertices) == 0:
        obj_eval.to_mesh_clear()
        raise RuntimeError(f"Mesh '{obj.name}' has no vertices after evaluation")
    return obj_eval, mesh


def release_evaluated_mesh(obj_eval, mesh):
    """Attempt to clear evaluated mesh safely."""
    try:
        obj_eval.to_mesh_clear()
    except Exception:
        pass  # intentional: try:


def get_bbox_local(obj):
    """Return list of 8 corner Vectors in object local space (from bound_box)."""
    return [Vector(corner) for corner in obj.bound_box]


# Object types whose bound_box describes real geometry extents. Everything else
# (empties, lights, cameras, speakers, lattices) has no meaningful volume, so we
# treat its origin point as its only contributing location.
GEOMETRY_TYPES = {
    'MESH', 'CURVE', 'SURFACE', 'META', 'FONT',
    'CURVES', 'POINTCLOUD', 'VOLUME', 'GPENCIL', 'GREASEPENCIL',
}


def obj_world_points(obj):
    """Return world-space points that bound an object, for group/center math.

    Geometry objects contribute their 8 world bbox corners; non-geometry objects
    (empties, lights, cameras, etc.) contribute only their world origin. This is
    what lets the group/proportional tools accept non-mesh objects without
    crashing on a degenerate bounding box.
    """
    if obj.type in GEOMETRY_TYPES:
        return [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return [obj.matrix_world.translation.copy()]


VALID_FRONT_AXES = {
    'LOCAL_Y_POS', 'LOCAL_Y_NEG',
    'LOCAL_X_POS', 'LOCAL_X_NEG',
    'LOCAL_Z_POS', 'LOCAL_Z_NEG',
}

def get_obj_front_axis(obj, scene=None):
    """
    Return the committed front axis for obj if one exists,
    otherwise +Y (Blender default). Does NOT fall back to scene setting —
    that's intentional to avoid circular reads when syncing the dropdown.
    """
    val = obj.get('radixpro_front_axis') if obj else None
    if val in VALID_FRONT_AXES:
        return val
    return 'LOCAL_Y_POS'


def get_front_axis_params(scene, bbox_verts_local):
    """
    Return (front_val_local, quad_builder_fn, side_key_fn) based on scene.origin_front_axis.

    Winding convention: vertices are ordered so the face normal points OUTWARD
    (away from the object centre) for every axis. This ensures apply_quad_offset
    always pushes the highlight away from the mesh surface, not into it.

    Outward normal check per face:
      +Y face (max Y): normal must be  0,+1, 0  → winding: (-x,+z) → (+x,+z) → (+x,-z) → (-x,-z)  ✓
      -Y face (min Y): normal must be  0,-1, 0  → reverse winding of +Y
      +X face (max X): normal must be +1, 0, 0  → reverse winding of naive -X result
      -X face (min X): normal must be -1, 0, 0  → winding: (+y,+z) → (-y,+z) → ...  ✓ as-was
      +Z face (max Z): normal must be  0, 0,+1
      -Z face (min Z): normal must be  0, 0,-1
    """
    axis = scene.origin_front_axis

    if axis == 'LOCAL_Y_POS':
        front_val = max(v.y for v in bbox_verts_local)
        def quad_fn(fx_min, fx_max, fz_min, fz_max, fv):
            # Normal = 0,+1,0 ✓
            return [
                Vector((fx_min, fv, fz_max)),
                Vector((fx_max, fv, fz_max)),
                Vector((fx_max, fv, fz_min)),
                Vector((fx_min, fv, fz_min)),
            ]
        side_key = lambda v: v.x

    elif axis == 'LOCAL_Y_NEG':
        front_val = min(v.y for v in bbox_verts_local)
        def quad_fn(fx_min, fx_max, fz_min, fz_max, fv):
            # Reversed winding → normal = 0,-1,0 (outward for min-Y face) ✓
            return [
                Vector((fx_max, fv, fz_max)),
                Vector((fx_min, fv, fz_max)),
                Vector((fx_min, fv, fz_min)),
                Vector((fx_max, fv, fz_min)),
            ]
        side_key = lambda v: v.x

    elif axis == 'LOCAL_X_POS':
        front_val = max(v.x for v in bbox_verts_local)
        def quad_fn(fx_min, fx_max, fz_min, fz_max, fv):
            # fx_min/max = Y extents. Winding → normal = +1,0,0 ✓
            return [
                Vector((fv, fx_max, fz_max)),
                Vector((fv, fx_min, fz_max)),
                Vector((fv, fx_min, fz_min)),
                Vector((fv, fx_max, fz_min)),
            ]
        side_key = lambda v: v.y

    elif axis == 'LOCAL_X_NEG':
        front_val = min(v.x for v in bbox_verts_local)
        def quad_fn(fx_min, fx_max, fz_min, fz_max, fv):
            # Normal = -1,0,0 ✓
            return [
                Vector((fv, fx_min, fz_max)),
                Vector((fv, fx_max, fz_max)),
                Vector((fv, fx_max, fz_min)),
                Vector((fv, fx_min, fz_min)),
            ]
        side_key = lambda v: v.y

    elif axis == 'LOCAL_Z_POS':
        front_val = max(v.z for v in bbox_verts_local)
        def quad_fn(fx_min, fx_max, fz_min, fz_max, fv):
            # fx = X extents, fz = Y extents. Normal = 0,0,+1 ✓
            return [
                Vector((fx_min, fz_min, fv)),
                Vector((fx_max, fz_min, fv)),
                Vector((fx_max, fz_max, fv)),
                Vector((fx_min, fz_max, fv)),
            ]
        side_key = lambda v: v.x

    else:  # LOCAL_Z_NEG
        front_val = min(v.z for v in bbox_verts_local)
        def quad_fn(fx_min, fx_max, fz_min, fz_max, fv):
            # Reversed → normal = 0,0,-1 ✓
            return [
                Vector((fx_min, fz_max, fv)),
                Vector((fx_max, fz_max, fv)),
                Vector((fx_max, fz_min, fv)),
                Vector((fx_min, fz_min, fv)),
            ]
        side_key = lambda v: v.x

    return front_val, quad_fn, side_key




def local_to_world(obj, v_local):
    """Local -> world using object's matrix_world."""
    return obj.matrix_world @ Vector(v_local)


def avg_vec(vecs):
    if not vecs:
        return Vector((0.0, 0.0, 0.0))
    s = Vector((0.0, 0.0, 0.0))
    for v in vecs:
        s += v
    return s / len(vecs)


def edge_midpoint_extreme(obj, axis1, extreme1, axis2, extreme2):
    """
    Compute the midpoint of vertices that lie at the intersection of two extremes.
    For example: axis1='Z', extreme1='max', axis2='X', extreme2='max' gives the 
    top-right edge midpoint.
    
    Returns world-space Vector or None.
    """
    obj_eval, mesh = get_evaluated_mesh(obj)
    try:
        verts_local = [v.co.copy() for v in mesh.vertices]
        mw = obj_eval.matrix_world
        
        if not verts_local:
            return None
        
        # Get extreme values for each axis
        def get_extreme_val(axis, extreme):
            if axis == 'X':
                return max(v.x for v in verts_local) if extreme == 'max' else min(v.x for v in verts_local)
            elif axis == 'Y':
                return max(v.y for v in verts_local) if extreme == 'max' else min(v.y for v in verts_local)
            else:  # 'Z'
                return max(v.z for v in verts_local) if extreme == 'max' else min(v.z for v in verts_local)
        
        def get_axis_val(v, axis):
            if axis == 'X':
                return v.x
            elif axis == 'Y':
                return v.y
            else:
                return v.z
        
        # Calculate tolerances based on object dimensions
        dims = [
            max(v.x for v in verts_local) - min(v.x for v in verts_local),
            max(v.y for v in verts_local) - min(v.y for v in verts_local),
            max(v.z for v in verts_local) - min(v.z for v in verts_local)
        ]
        tol = max(dims) * 1e-4 + 1e-6
        
        extreme_val1 = get_extreme_val(axis1, extreme1)
        extreme_val2 = get_extreme_val(axis2, extreme2)
        
        # Find vertices at both extremes (the edge vertices)
        edge_verts = []
        for v in verts_local:
            val1 = get_axis_val(v, axis1)
            val2 = get_axis_val(v, axis2)
            
            if abs(val1 - extreme_val1) < tol and abs(val2 - extreme_val2) < tol:
                edge_verts.append(v)
        
        if not edge_verts:
            return None
        
        # Average the edge vertices in local space, then convert to world
        midpoint_local = avg_vec(edge_verts)
        return mw @ midpoint_local
        
    finally:
        release_evaluated_mesh(obj_eval, mesh)


def polygon_area_from_world_coords(world_vs):
    """Area of polygon from world-space vertex list (triangulate fan)."""
    if len(world_vs) < 3:
        return 0.0
    area = 0.0
    v0 = world_vs[0]
    for i in range(1, len(world_vs) - 1):
        a = world_vs[i] - v0
        b = world_vs[i + 1] - v0
        area += a.cross(b).length / 2.0
    return area


# ---------------- Geometry-driven face centroid helpers (local-axis aware) ----------------
def face_centroid_extreme(obj, axis, extreme='max'):
    """
    Compute an area-weighted centroid (in world space) of polygons whose vertices lie at the
    extreme along a given LOCAL axis. Returns a world-space Vector or None.
    """
    obj_eval, mesh = get_evaluated_mesh(obj)
    try:
        verts_local = [v.co.copy() for v in mesh.vertices]
        mw = obj_eval.matrix_world
        polys = list(mesh.polygons)

        if not verts_local or not polys:
            return None

        # compute local extremes
        if axis == 'X':
            extreme_val = max(v.x for v in verts_local) if extreme == 'max' else min(v.x for v in verts_local)
            tol_basis = max((max(v.y for v in verts_local) - min(v.y for v in verts_local)),
                            (max(v.z for v in verts_local) - min(v.z for v in verts_local)), 1e-6)
        elif axis == 'Y':
            extreme_val = max(v.y for v in verts_local) if extreme == 'max' else min(v.y for v in verts_local)
            tol_basis = max((max(v.x for v in verts_local) - min(v.x for v in verts_local)),
                            (max(v.z for v in verts_local) - min(v.z for v in verts_local)), 1e-6)
        else:  # 'Z'
            extreme_val = max(v.z for v in verts_local) if extreme == 'max' else min(v.z for v in verts_local)
            tol_basis = max((max(v.x for v in verts_local) - min(v.x for v in verts_local)),
                            (max(v.y for v in verts_local) - min(v.y for v in verts_local)), 1e-6)

        tol = tol_basis * 1e-4 + 1e-6

        total_area = 0.0
        weighted_centroid = Vector((0.0, 0.0, 0.0))
        for poly in polys:
            poly_local_vs = [verts_local[i] for i in poly.vertices]
            all_at_extreme = True
            for lv in poly_local_vs:
                val = lv.x if axis == 'X' else (lv.y if axis == 'Y' else lv.z)
                if abs(val - extreme_val) > tol:
                    all_at_extreme = False
                    break
            if not all_at_extreme:
                continue

            world_vs = [mw @ lv for lv in poly_local_vs]
            area = polygon_area_from_world_coords(world_vs)
            if area <= EPS:
                area = 1.0
            center = avg_vec(world_vs)
            weighted_centroid += center * area
            total_area += area

        if total_area > 0.0:
            return weighted_centroid / total_area

        # fallback
        tol2 = tol_basis * 1e-3 + 1e-6
        if axis == 'X':
            layer_local = [lv for lv in verts_local if abs(lv.x - extreme_val) < tol2]
        elif axis == 'Y':
            layer_local = [lv for lv in verts_local if abs(lv.y - extreme_val) < tol2]
        else:
            layer_local = [lv for lv in verts_local if abs(lv.z - extreme_val) < tol2]

        if layer_local:
            layer_world = [mw @ lv for lv in layer_local]
            return avg_vec(layer_world)

        return None
    finally:
        release_evaluated_mesh(obj_eval, mesh)


# ---------------- Multi-Object Combined Extremes ----------------
def get_combined_face_centroid(objects, axis, extreme='max', source='MESH'):
    """
    Calculate area-weighted face centroid across ALL objects' combined mesh in WORLD space.
    """
    # STEP 1: Find the GLOBAL extreme value across all objects in WORLD coordinates
    global_extreme_val = None

    # We'll also collect per-object world verts/polygons to reuse later
    obj_world_data = []

    for obj in objects:
        if obj.type != 'MESH':
            continue

        if source == 'MESH':
            try:
                obj_eval, mesh = get_evaluated_mesh(obj)
                try:
                    mw = obj_eval.matrix_world
                    # world-space coords for verts
                    verts_world = [mw @ v.co for v in mesh.vertices]
                    polys = list(mesh.polygons)
                    if not verts_world or not polys:
                        continue

                    # Get extreme in world space
                    if axis == 'X':
                        obj_extreme = max(v.x for v in verts_world) if extreme == 'max' else min(v.x for v in verts_world)
                    elif axis == 'Y':
                        obj_extreme = max(v.y for v in verts_world) if extreme == 'max' else min(v.y for v in verts_world)
                    else:  # 'Z'
                        obj_extreme = max(v.z for v in verts_world) if extreme == 'max' else min(v.z for v in verts_world)

                    # Update global extreme
                    if global_extreme_val is None:
                        global_extreme_val = obj_extreme
                    else:
                        global_extreme_val = max(global_extreme_val, obj_extreme) if extreme == 'max' else min(global_extreme_val, obj_extreme)

                    obj_world_data.append((obj_eval, mesh, verts_world, polys, mw))
                finally:
                    # do NOT clear here – we'll release later after we use meshes in step 2
                    pass
            except Exception as e:
                print(f"[Origin Snap] Error processing object {obj.name}: {e}")
                continue
        else:
            # BBox source: use bbox corners converted to world space
            try:
                bbox_local = get_bbox_local(obj)
                bbox_world = [obj.matrix_world @ v for v in bbox_local]
                if not bbox_world:
                    continue
                if axis == 'X':
                    obj_extreme = max(v.x for v in bbox_world) if extreme == 'max' else min(v.x for v in bbox_world)
                elif axis == 'Y':
                    obj_extreme = max(v.y for v in bbox_world) if extreme == 'max' else min(v.y for v in bbox_world)
                else:
                    obj_extreme = max(v.z for v in bbox_world) if extreme == 'max' else min(v.z for v in bbox_world)

                if global_extreme_val is None:
                    global_extreme_val = obj_extreme
                else:
                    global_extreme_val = max(global_extreme_val, obj_extreme) if extreme == 'max' else min(global_extreme_val, obj_extreme)

                # For BBOX source emulate simple rectangular faces using bbox_world as polys
                obj_world_data.append((None, None, bbox_world, None, obj.matrix_world))
            except Exception as e:
                print(f"[Origin Snap] Error processing bbox for {obj.name}: {e}")
                continue

    if global_extreme_val is None:
        # nothing found
        # release any evaluated meshes we kept open
        for d in obj_world_data:
            obj_eval = d[0]
            mesh = d[1]
            if obj_eval and mesh:
                try:
                    obj_eval.to_mesh_clear()
                except Exception:
                    pass  # intentional: try:
        return None

    # STEP 2: Now collect faces that lie at this GLOBAL extreme (use world-space)
    total_area = 0.0
    weighted_centroid = Vector((0.0, 0.0, 0.0))
    fallback_verts = []  # Collect all extreme vertices as fallback

    for item in obj_world_data:
        obj_eval, mesh, verts_world, polys, mw = item
        try:
            if mesh and polys:
                # compute tolerance in world space based on object extents
                xs = [v.x for v in verts_world]
                ys = [v.y for v in verts_world]
                zs = [v.z for v in verts_world]
                if axis == 'X':
                    tol_basis = max(max(ys) - min(ys), max(zs) - min(zs), 1e-6)
                elif axis == 'Y':
                    tol_basis = max(max(xs) - min(xs), max(zs) - min(zs), 1e-6)
                else:
                    tol_basis = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
                tol = tol_basis * 1e-3 + 1e-6

                # Collect vertices at extreme for fallback
                for v in verts_world:
                    val = v.x if axis == 'X' else (v.y if axis == 'Y' else v.z)
                    if abs(val - global_extreme_val) < tol:
                        fallback_verts.append(v)

                for poly in polys:
                    poly_world_vs = [verts_world[i] for i in poly.vertices]
                    all_at_extreme = True
                    for wv in poly_world_vs:
                        val = wv.x if axis == 'X' else (wv.y if axis == 'Y' else wv.z)
                        if abs(val - global_extreme_val) > tol:
                            all_at_extreme = False
                            break
                    if not all_at_extreme:
                        continue

                    area = polygon_area_from_world_coords(poly_world_vs)
                    if area <= EPS:
                        area = 1.0
                    center = avg_vec(poly_world_vs)
                    weighted_centroid += center * area
                    total_area += area
            else:
                # bbox-only fallback (treat bbox corners as a fake face layer)
                if verts_world:
                    # pick verts close to global extreme
                    if axis == 'X':
                        layer_vs = [v for v in verts_world if abs(v.x - global_extreme_val) < 1e-5]
                    elif axis == 'Y':
                        layer_vs = [v for v in verts_world if abs(v.y - global_extreme_val) < 1e-5]
                    else:
                        layer_vs = [v for v in verts_world if abs(v.z - global_extreme_val) < 1e-5]

                    if layer_vs:
                        fallback_verts.extend(layer_vs)
                        center = avg_vec(layer_vs)
                        weighted_centroid += center * 1.0
                        total_area += 1.0
        except Exception as e:
            print(f"[Origin Snap] Error calculating centroid: {e}")
            continue
        finally:
            # release evaluated mesh if any
            if obj_eval and mesh:
                try:
                    obj_eval.to_mesh_clear()
                except Exception:
                    pass  # intentional: try:

    if total_area > 0.0:
        return weighted_centroid / total_area

    # FALLBACK: If no faces found, use average of extreme vertices
    if fallback_verts:
        return avg_vec(fallback_verts)

    return None

def get_combined_extreme_vertex(objects, ideal_local, source='MESH'):
    """
    Option B: Find single most extreme vertex across all objects.
    Used for Centers, Vertices, and Edges buttons.
    """
    all_verts_world = []

    for obj in objects:
        if obj.type != 'MESH':
            continue

        if source == 'MESH':
            obj_eval, mesh = get_evaluated_mesh(obj)
            try:
                verts_world = [obj_eval.matrix_world @ v.co for v in mesh.vertices]
                all_verts_world.extend(verts_world)
            finally:
                release_evaluated_mesh(obj_eval, mesh)
        else:
            bbox_local = get_bbox_local(obj)
            verts_world = [obj.matrix_world @ v for v in bbox_local]
            all_verts_world.extend(verts_world)

    if not all_verts_world:
        return None

    # Find vertex closest to ideal position
    if objects:
        ideal_world = objects[0].matrix_world @ ideal_local
    else:
        ideal_world = ideal_local

    best_vert = None
    best_dist = None
    for v in all_verts_world:
        dist = (v - ideal_world).length
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_vert = v

    return best_vert


# ---------------- Closest vertex helper (local-space based) ----------------
def find_closest_vertex_index_local(verts_local, ideal_local):
    """Return index of verts_local that is closest to ideal_local (in local coords)."""
    if not verts_local:
        return None
    best_i = None
    best_d2 = None
    for i, v in enumerate(verts_local):
        d2 = (v - ideal_local).length_squared
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


# ---------------- Cleanup orphaned previews ----------------
def cleanup_deleted_objects():
    """Remove previews for objects no longer in the scene AND re-key renamed objects.

    Two cases handled:
    1. DELETED: key is not in bpy.data.objects at all -> evict from cache.
    2. RENAMED: a cached object still exists but its name changed. Blender renames
       the object in-place, so obj.name differs from the cache key. We re-key the
       cache entry to the new name so the draw handler still finds it.
       Without this, renaming an object while preview is on produces a ghost quad
       that never clears.
    """
    global _persistent_previews

    valid_names = {obj.name for obj in bpy.data.objects if obj.type == 'MESH'}
    orphaned = []
    renames  = {}  # old_key -> new_key

    for key in list(_persistent_previews.keys()):
        if key in valid_names:
            continue
        # May be a rename: if an object with this key still physically exists
        # but Blender already updated its .name, the stashed _obj_ref will
        # reveal the new name.
        obj_ref = _persistent_previews[key].get('_obj_ref')
        if obj_ref is not None:
            try:
                new_name = obj_ref.name  # ReferenceError if deleted
                if new_name != key and new_name in valid_names:
                    renames[key] = new_name
                    continue
            except ReferenceError:
                pass
        orphaned.append(key)

    for old_key, new_key in renames.items():
        _persistent_previews[new_key] = _persistent_previews.pop(old_key)

    for obj_name in orphaned:
        data = _persistent_previews.pop(obj_name, {})
        arrow = data.get('arrow')
        if arrow:
            try:
                if arrow.name in bpy.data.objects:
                    for col in list(arrow.users_collection):
                        try:
                            col.objects.unlink(arrow)
                        except Exception:
                            pass
                    try:
                        bpy.data.objects.remove(arrow, do_unlink=True)
                    except Exception:
                        pass
            except ReferenceError:
                pass

    return bool(orphaned) or bool(renames)


# ── Deferred cleanup (NEVER mutate blend data from a draw or depsgraph handler) ──
# Removing objects from a POST_VIEW draw callback can corrupt state or crash
# Blender. cleanup_deleted_objects() therefore must only ever run from a timer,
# which executes in a context where bpy.data mutation is safe.
_cleanup_scheduled = False


def _tag_view3d_redraw():
    """Tag all 3D viewports for redraw (robust to a missing context.screen)."""
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


def _deferred_cleanup():
    """Timer callback: run orphan-preview cleanup in a data-safe context."""
    global _cleanup_scheduled
    _cleanup_scheduled = False
    try:
        if cleanup_deleted_objects():
            _tag_view3d_redraw()
    except Exception as e:
        print(f"[Radix] deferred cleanup failed: {e}")
    return None  # one-shot


def schedule_cleanup():
    """Request a deferred cleanup. Coalesces many requests into a single timer,
    so calling this every draw frame costs at most one pending timer."""
    global _cleanup_scheduled
    if _cleanup_scheduled:
        return
    _cleanup_scheduled = True
    try:
        bpy.app.timers.register(_deferred_cleanup, first_interval=0.0)
    except Exception:
        _cleanup_scheduled = False

def remove_preview_objects():
    """Clear preview state. GPU-drawn arrows need no bpy.data cleanup.
    Also sweeps for any legacy SINGLE_ARROW empties left by older versions."""
    global _preview_cache, _persistent_previews

    _preview_cache.clear()
    _persistent_previews.clear()

    # Sweep for legacy empties (from versions before GPU-drawn arrows)
    legacy = [o for o in bpy.data.objects
              if o.name.startswith(PREVIEW_ARROW_NAME)]
    for o in legacy:
        for col in list(o.users_collection):
            try: col.objects.unlink(o)
            except Exception: pass
        try: bpy.data.objects.remove(o, do_unlink=True)
        except Exception: pass

    # Remove the preview collection if it exists and is now empty
    col = bpy.data.collections.get(PREVIEW_COLLECTION)
    if col:
        if len(col.objects) == 0:
            try: bpy.context.scene.collection.children.unlink(col)
            except Exception: pass
            try: bpy.data.collections.remove(col)
            except Exception: pass


# ---------------- GPU draw handler ----------------
def draw_combined_bbox_wireframe(scene):
    """Draw wireframe for bounding boxes when BBox mode is active"""
    
    # Check toggle first
    if not scene.origin_show_bbox_wireframe:
        return
    
    # Only draw in BBOX mode
    if scene.origin_snap_source != 'BBOX':
        return
    
    # Determine which objects to draw based on Multi-Object mode
    if scene.origin_multi_object_preview:
        # Multi-Object mode: draw for all selected meshes
        target_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if len(target_objects) < 1:
            return
        
        # For COMBINED mode with 2+ objects: draw one wireframe for all
        # For INDIVIDUAL mode or single selection: draw separate wireframes
        if scene.origin_mo_snap_mode == 'COMBINED' and len(target_objects) >= 2:
            # Calculate combined bounding box
            all_bbox_corners_world = []
            for obj in target_objects:
                bbox_local = get_bbox_local(obj)
                bbox_world = [obj.matrix_world @ v for v in bbox_local]
                all_bbox_corners_world.extend(bbox_world)
            
            if not all_bbox_corners_world:
                return
            
            # Get combined bbox extremes
            min_x = min(v.x for v in all_bbox_corners_world)
            max_x = max(v.x for v in all_bbox_corners_world)
            min_y = min(v.y for v in all_bbox_corners_world)
            max_y = max(v.y for v in all_bbox_corners_world)
            min_z = min(v.z for v in all_bbox_corners_world)
            max_z = max(v.z for v in all_bbox_corners_world)
            
            # Draw single wireframe for combined bbox
            draw_bbox_wireframe_cube(scene, min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            # INDIVIDUAL mode: draw separate wireframe for each object
            for obj in target_objects:
                bbox_local = get_bbox_local(obj)
                if not bbox_local:
                    continue
                bbox_world = [obj.matrix_world @ v for v in bbox_local]
                draw_oriented_bbox_wireframe(scene, bbox_world)
    else:
        # Single-Object mode: draw for active object only
        obj = bpy.context.active_object
        if not obj or obj.type != 'MESH':
            return
        
        bbox_local = get_bbox_local(obj)
        if not bbox_local:
            return
        
        # Transform bbox corners to world space (preserves rotation!)
        bbox_world = [obj.matrix_world @ v for v in bbox_local]
        
        # Draw oriented wireframe
        draw_oriented_bbox_wireframe(scene, bbox_world)


def draw_oriented_bbox_wireframe(scene, bbox_corners):
    """Draw oriented bounding box wireframe using 8 transformed corners"""
    
    if len(bbox_corners) != 8:
        return
    
    # Define 12 edges of the cube (using local bbox corner indices)
    # bbox_corners are already in world space with rotation/scale applied
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # Bottom face
        (4,5), (5,6), (6,7), (7,4),  # Top face
        (0,4), (1,5), (2,6), (3,7),  # Vertical edges
    ]
    
    # Create line coordinates from transformed corners
    coords = []
    for start, end in edges:
        coords.append(bbox_corners[start])
        coords.append(bbox_corners[end])
    
    # Draw wireframe
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        
        color = scene.origin_preview_color
        alpha = 0.7  # Slightly more opaque than highlight
        
        gpu.state.line_width_set(2.0)
        shader.bind()
        shader.uniform_float("color", (*color, alpha))
        batch.draw(shader)
        gpu.state.line_width_set(1.0)
    except Exception as e:
        print(f"[Origin Snap] Failed to draw oriented bbox wireframe: {e}")


def draw_bbox_wireframe_cube(scene, min_x, max_x, min_y, max_y, min_z, max_z):
    """Helper function to draw a single bbox wireframe cube"""
    
    # Create 8 corners of bbox
    combined_bbox = [
        Vector((min_x, min_y, min_z)),  # 0
        Vector((max_x, min_y, min_z)),  # 1
        Vector((max_x, max_y, min_z)),  # 2
        Vector((min_x, max_y, min_z)),  # 3
        Vector((min_x, min_y, max_z)),  # 4
        Vector((max_x, min_y, max_z)),  # 5
        Vector((max_x, max_y, max_z)),  # 6
        Vector((min_x, max_y, max_z)),  # 7
    ]
    
    # Define 12 edges of the cube
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # Bottom face
        (4,5), (5,6), (6,7), (7,4),  # Top face
        (0,4), (1,5), (2,6), (3,7),  # Vertical edges
    ]
    
    # Create line coordinates
    coords = []
    for start, end in edges:
        coords.append(combined_bbox[start])
        coords.append(combined_bbox[end])
    
    # Draw wireframe
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        
        color = scene.origin_preview_color
        alpha = 0.7  # Slightly more opaque than highlight
        
        gpu.state.line_width_set(2.0)
        shader.bind()
        shader.uniform_float("color", (*color, alpha))
        batch.draw(shader)
        gpu.state.line_width_set(1.0)
    except Exception as e:
        print(f"[Origin Snap] BBox wireframe draw failed: {e}")


def apply_quad_offset(quad, offset_distance):
    """Push quad vertices forward along face normal by offset_distance"""
    if not quad or len(quad) < 3 or offset_distance == 0:
        return quad
    
    # Calculate face normal from first 3 vertices (counter-clockwise winding)
    v1 = Vector(quad[0])
    v2 = Vector(quad[1])
    v3 = Vector(quad[2])
    
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = edge1.cross(edge2).normalized()
    
    # Offset all vertices along normal
    return [Vector(v) + normal * offset_distance for v in quad]


def quad_local_to_world(quad_local, bbox_local, orientation, front_axis, obj):
    """Re-derive world-space quad from stored local data every draw frame.
    
    LOCAL orientation: quad_local holds 4 verts in object space → apply matrix_world.
    GLOBAL orientation: quad_local holds 8 bbox verts in object space → recompute
                        the world-axis-aligned quad using the current matrix_world.
    """
    mw = obj.matrix_world

    if orientation == 'LOCAL':
        return [mw @ v for v in quad_local]

    # GLOBAL: rebuild world quad from current bbox
    from mathutils import Vector as _V
    bbox_world = [mw @ v for v in bbox_local]
    tol = 1e-5
    fa  = front_axis

    def _quad(axis_fn, extreme_fn, side_a, side_b, vert_fn):
        fv = extreme_fn(axis_fn(v) for v in bbox_world)
        front = [v for v in bbox_world if abs(axis_fn(v) - fv) < tol] or bbox_world
        a_min = min(side_a(v) for v in front); a_max = max(side_a(v) for v in front)
        b_min = min(side_b(v) for v in front); b_max = max(side_b(v) for v in front)
        return vert_fn(fv, a_min, a_max, b_min, b_max)

    if fa == 'LOCAL_Y_POS':
        fv = max(v.y for v in bbox_world)
        fw = [v for v in bbox_world if abs(v.y - fv) < tol] or bbox_world
        return [_V((min(v.x for v in fw), fv, max(v.z for v in fw))),
                _V((max(v.x for v in fw), fv, max(v.z for v in fw))),
                _V((max(v.x for v in fw), fv, min(v.z for v in fw))),
                _V((min(v.x for v in fw), fv, min(v.z for v in fw)))]
    elif fa == 'LOCAL_Y_NEG':
        fv = min(v.y for v in bbox_world)
        fw = [v for v in bbox_world if abs(v.y - fv) < tol] or bbox_world
        return [_V((max(v.x for v in fw), fv, max(v.z for v in fw))),
                _V((min(v.x for v in fw), fv, max(v.z for v in fw))),
                _V((min(v.x for v in fw), fv, min(v.z for v in fw))),
                _V((max(v.x for v in fw), fv, min(v.z for v in fw)))]
    elif fa == 'LOCAL_X_POS':
        fv = max(v.x for v in bbox_world)
        fw = [v for v in bbox_world if abs(v.x - fv) < tol] or bbox_world
        return [_V((fv, max(v.y for v in fw), max(v.z for v in fw))),
                _V((fv, min(v.y for v in fw), max(v.z for v in fw))),
                _V((fv, min(v.y for v in fw), min(v.z for v in fw))),
                _V((fv, max(v.y for v in fw), min(v.z for v in fw)))]
    elif fa == 'LOCAL_X_NEG':
        fv = min(v.x for v in bbox_world)
        fw = [v for v in bbox_world if abs(v.x - fv) < tol] or bbox_world
        return [_V((fv, min(v.y for v in fw), max(v.z for v in fw))),
                _V((fv, max(v.y for v in fw), max(v.z for v in fw))),
                _V((fv, max(v.y for v in fw), min(v.z for v in fw))),
                _V((fv, min(v.y for v in fw), min(v.z for v in fw)))]
    elif fa == 'LOCAL_Z_POS':
        fv = max(v.z for v in bbox_world)
        fw = [v for v in bbox_world if abs(v.z - fv) < tol] or bbox_world
        return [_V((min(v.x for v in fw), min(v.y for v in fw), fv)),
                _V((max(v.x for v in fw), min(v.y for v in fw), fv)),
                _V((max(v.x for v in fw), max(v.y for v in fw), fv)),
                _V((min(v.x for v in fw), max(v.y for v in fw), fv))]
    else:  # LOCAL_Z_NEG
        fv = min(v.z for v in bbox_world)
        fw = [v for v in bbox_world if abs(v.z - fv) < tol] or bbox_world
        return [_V((min(v.x for v in fw), max(v.y for v in fw), fv)),
                _V((max(v.x for v in fw), max(v.y for v in fw), fv)),
                _V((max(v.x for v in fw), min(v.y for v in fw), fv)),
                _V((min(v.x for v in fw), min(v.y for v in fw), fv))]

def _make_shape_tris(shape, cx, cy, r):
    """Return a flat list of (x, y) tuples representing the shape as filled triangles.

    All shapes are centered at (cx, cy) with approximate radius r.
    Shapes are generated at their natural orientation:
      DIAMOND   — corners       — orange  — 4 equidistant points, rotated 45°
      SQUARE    — edge mids     — red     — axis-aligned square
      TRIANGLE  — face centers  — blue    — equilateral pointing up
      CIRCLE    — bbox center   — yellow  — 12-segment polygon
    """
    import math
    if shape == 'DIAMOND':
        top   = (cx,     cy + r)
        right = (cx + r, cy    )
        bot   = (cx,     cy - r)
        left  = (cx - r, cy    )
        return [top, right, bot,   top, bot, left]

    elif shape == 'SQUARE':
        s = r * 0.82
        tl = (cx - s, cy + s); tr = (cx + s, cy + s)
        bl = (cx - s, cy - s); br = (cx + s, cy - s)
        return [tl, tr, br,   tl, br, bl]

    elif shape == 'TRIANGLE':
        s60 = math.sin(math.radians(60))
        top = (cx,            cy + r)
        bl  = (cx - r * s60, cy - r * 0.5)
        br  = (cx + r * s60, cy - r * 0.5)
        return [top, br, bl]

    elif shape == 'CIRCLE':
        verts, n = [], 14
        for i in range(n):
            a1 = 2 * math.pi * i       / n
            a2 = 2 * math.pi * (i + 1) / n
            verts += [
                (cx, cy),
                (cx + r * math.cos(a1), cy + r * math.sin(a1)),
                (cx + r * math.cos(a2), cy + r * math.sin(a2)),
            ]
        return verts

    return []


def _draw_single_handle(shader, cx, cy, shape, fill_color, size):
    """Draw one handle: black outline then colored fill for contrast on any background."""
    # Outer black shadow (contrast against bright geometry)
    outer = _make_shape_tris(shape, cx, cy, size + 2.8)
    if outer:
        b = batch_for_shader(shader, 'TRIS', {"pos": outer})
        shader.bind()
        shader.uniform_float("color", (0.0, 0.0, 0.0, 0.85))
        b.draw(shader)

    # Thin white ring for contrast against dark backgrounds
    white = _make_shape_tris(shape, cx, cy, size + 1.2)
    if white:
        b = batch_for_shader(shader, 'TRIS', {"pos": white})
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.65))
        b.draw(shader)

    # Color fill
    fill = _make_shape_tris(shape, cx, cy, size)
    if fill:
        b = batch_for_shader(shader, 'TRIS', {"pos": fill})
        shader.bind()
        shader.uniform_float("color", (*fill_color, 1.0))
        b.draw(shader)


def draw_handle_shapes_2d():
    """POST_PIXEL callback: draw shaped, colored, outlined bbox handles in screen space.

    Runs in 2D screen space so handle shapes are a FIXED PIXEL SIZE regardless of
    zoom level or object distance — matching the legibility of Blender's own gizmos.

    Shape ↔ handle type:
      Diamond  (orange) → bbox corners   (8 handles)
      Square   (red)    → edge midpoints (12 handles)
      Triangle (blue)   → face centers   (6 handles)
      Circle   (yellow) → bbox center    (1 handle, only in BBOX_CENTER mode)
    """
    try:
        scene = bpy.context.scene
    except Exception:
        return

    if not getattr(scene, 'origin_show_viewport_handles', False):
        return

    try:
        obj = bpy.context.active_object
    except Exception:
        return

    if not obj or obj.type not in GEOMETRY_TYPES:
        return

    try:
        region = bpy.context.region
        rv3d   = bpy.context.space_data.region_3d if bpy.context.space_data else None
        if not region or not rv3d:
            return
    except Exception:
        return

    from bpy_extras.view3d_utils import location_3d_to_region_2d
    mode = getattr(scene, 'origin_handles_snap_type', 'ALL')

    # ── Compute world-space handle positions ───────────────────────────────────
    mw  = obj.matrix_world
    bbl = [Vector(c) for c in obj.bound_box]
    bbw = [mw @ v for v in bbl]

    xs = [v.x for v in bbw]; ys = [v.y for v in bbw]; zs = [v.z for v in bbw]
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    md = (mn + mx) * 0.5

    # (world_pos, shape, fill_rgba_3)
    handles = []

    if mode in ('CORNERS', 'ALL'):
        for v in bbw:
            handles.append((v.copy(), 'DIAMOND', (0.96, 0.59, 0.18)))  # orange

    if mode in ('EDGE_MIDPOINTS', 'ALL'):
        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for a, b in edges:
            mid = (bbw[a] + bbw[b]) * 0.5
            handles.append((mid, 'SQUARE', (0.90, 0.13, 0.12)))  # red

    if mode in ('FACE_CENTERS', 'ALL'):
        face_centers = [
            Vector((md.x, md.y, mx.z)),  # top
            Vector((md.x, md.y, mn.z)),  # bottom
            Vector((md.x, mx.y, md.z)),  # front
            Vector((md.x, mn.y, md.z)),  # back
            Vector((mx.x, md.y, md.z)),  # right
            Vector((mn.x, md.y, md.z)),  # left
        ]
        for pos in face_centers:
            handles.append((pos, 'TRIANGLE', (0.18, 0.55, 0.99)))  # blue

    if mode == 'BBOX_CENTER':
        handles.append((md.copy(), 'CIRCLE', (1.0, 0.90, 0.15)))  # yellow

    if not handles:
        return

    # ── Project to screen and draw ─────────────────────────────────────────────
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')   # always on top

    HANDLE_SIZE = max(getattr(scene, 'origin_handle_size', 8.5), 4.0)
    HOVER_SCALE = 1.55          # how much to grow on hover
    HOVER_GLOW_BOOST = 0.18    # subtle brightness lift on hovered handle

    # Hovered world position written by the click_bbox_handle modal
    hover_pos = _handle_hover_world_pos[0]

    for world_pos, shape, color in handles:
        screen_pos = location_3d_to_region_2d(region, rv3d, world_pos)
        if screen_pos is None:
            continue
        cx, cy = screen_pos.x, screen_pos.y
        # Clip to region bounds with a small margin so half-visible handles still draw
        if cx < -20 or cx > region.width + 20 or cy < -20 or cy > region.height + 20:
            continue

        # Hover: check whether the modal flagged this exact world position
        is_hovered = (hover_pos is not None and
                      (world_pos - hover_pos).length < 0.001)

        draw_size = HANDLE_SIZE * HOVER_SCALE if is_hovered else HANDLE_SIZE
        draw_color = (
            tuple(min(1.0, c + HOVER_GLOW_BOOST) for c in color)
            if is_hovered else color
        )
        _draw_single_handle(shader, cx, cy, shape, draw_color, draw_size)

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('NONE')


def _draw_viewport_handles(scene, obj):
    """Legacy stub — actual drawing now happens in draw_handle_shapes_2d (POST_PIXEL).
    Kept so that any residual internal call sites don't error."""
    pass


def draw_overlay_3d():
    """POST_VIEW draw: translucent quad(s) with depth awareness.

    Quads are stored in local space and transformed to world every frame
    using the object's current matrix_world — so they follow object movement,
    rotation, and scale correctly without needing a depsgraph update trigger.
    """
    global _preview_cache
    global _persistent_previews

    try:
        scene = bpy.context.scene
    except Exception:
        return

    # Early exit — keeps the GPU path completely idle when preview is off.
    # Also prevents arrow mesh objects lingering after the toggle.
    if not getattr(scene, 'show_origin_preview', False):
        return

    # NEVER mutate blend data from a draw callback.
    if _persistent_previews and any(n not in bpy.data.objects for n in tuple(_persistent_previews)):
        schedule_cleanup()

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    gpu.state.blend_set('ALPHA')
    gpu.state.face_culling_set('NONE')

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    def _draw_quad(quad_world, color, alpha):
        """Draw a single face quad from world-space verts."""
        if not quad_world or len(quad_world) < 4:
            return
        offset_quad = apply_quad_offset(quad_world, scene.origin_face_offset)
        coords = [
            offset_quad[0][:], offset_quad[1][:], offset_quad[2][:],
            offset_quad[0][:], offset_quad[2][:], offset_quad[3][:]
        ]
        batch = batch_for_shader(shader, 'TRIS', {"pos": coords})
        shader.bind()
        shader.uniform_float("color", (color[0], color[1], color[2], alpha))
        batch.draw(shader)

    def _resolve_world_quad(quad_local, bbox_local, orientation, front_axis, obj):
        """Get live world-space quad using the object's current matrix_world.
        Restores the proven approach: quad_local holds 4 face verts in object
        space; mw @ each vert gives the correct world position every frame."""
        if obj is None or quad_local is None or bbox_local is None:
            return None
        try:
            return quad_local_to_world(quad_local, bbox_local, orientation, front_axis, obj)
        except Exception as e:
            print(f"[Radix] quad transform failed for '{obj.name}': {e}")
            return None

    # ── Active object preview ─────────────────────────────────────────────────
    cache_obj_name   = _preview_cache.get('obj_name')
    cache_quad_local = _preview_cache.get('quad_local')
    cache_bbox_local = _preview_cache.get('bbox_local')
    cache_orient     = _preview_cache.get('orientation')
    cache_axis       = _preview_cache.get('front_axis')

    if cache_obj_name and cache_quad_local and cache_bbox_local:
        try:
            obj = bpy.context.active_object
            # Blender keeps view_layer.objects.active set even after a full
            # deselect (Alt+A) — "active" and "selected" are independent
            # states. Require genuine selection so the highlight disappears
            # the moment nothing is selected, not just when active changes.
            if obj and obj.select_get() and obj.name == cache_obj_name and obj.type in GEOMETRY_TYPES:
                is_muted = _persistent_previews.get(obj.name, {}).get('muted', False)
                if not is_muted:
                    quad_world = _resolve_world_quad(
                        cache_quad_local, cache_bbox_local,
                        cache_orient, cache_axis, obj
                    )
                    if quad_world:
                        if mom_preview_active(scene):
                            color = get_collection_color(obj)
                        else:
                            color = scene.origin_preview_color
                        _draw_quad(quad_world, color, scene.origin_preview_alpha)
        except Exception as e:
            print(f"[Radix] draw failed for active object: {e}")

    # ── Persistent + Multi-Object previews ────────────────────────────────────
    draw_mo = mom_preview_active(scene)
    if scene.origin_keep_preview_persistent or draw_mo:
        active_name = getattr(bpy.context.active_object, 'name', None)
        # In pure MO mode (no persistence), only draw the currently-selected set;
        # persistent mode draws everything accumulated.
        sel_filter = None
        if draw_mo and not scene.origin_keep_preview_persistent:
            sel_filter = {o.name for o in bpy.context.selected_objects}

        for obj_name, data in list(_persistent_previews.items()):
            if obj_name == active_name:
                continue
            if sel_filter is not None and obj_name not in sel_filter:
                continue
            if data.get('muted', False):
                continue

            quad_local = data.get('quad_local')
            bbox_local = data.get('bbox_local')
            orient     = data.get('orientation')
            front_axis = data.get('front_axis')

            if not quad_local or not bbox_local:
                # Fallback for entries saved before local-space storage
                quad_world = data.get('quad')
                if not quad_world or len(quad_world) < 4:
                    continue
            else:
                try:
                    obj = bpy.data.objects.get(obj_name)
                    if obj is None or obj.type not in GEOMETRY_TYPES:
                        continue
                    quad_world = _resolve_world_quad(
                        quad_local, bbox_local, orient, front_axis, obj)
                    if not quad_world:
                        continue
                except Exception as e:
                    print(f"[Radix] persistent draw failed for '{obj_name}': {e}")
                    continue

            try:
                obj = bpy.data.objects.get(obj_name)
                cached_color = data.get('active_color')
                if cached_color is None:
                    if draw_mo and obj and obj.type in GEOMETRY_TYPES:
                        cached_color = get_collection_color(obj)
                    else:
                        cached_color = tuple(scene.origin_preview_color)
                    data['active_color'] = cached_color
                _draw_quad(quad_world, cached_color, scene.origin_preview_alpha)
            except Exception as e:
                print(f"[Radix] GPU draw failed for '{obj_name}': {e}")

    # ── Combined bbox wireframe ───────────────────────────────────────────────
    draw_combined_bbox_wireframe(scene)

    # ── GPU arrows (colour-grouped, live-computed) ────────────────────────────
    # Grouped by colour so MOM arrows use collection colours (matching the quads)
    # while keeping draw calls to O(unique colours) rather than O(objects).
    # Positions computed live from obj.matrix_world so arrows follow moves instantly.

    def _arrow_verts_live(obj, snap_local, scale_ratio, orientation):
        try:
            mw = obj.matrix_world
            if snap_local is not None:
                base = mw @ snap_local
            else:
                bb  = [Vector(c) for c in obj.bound_box]
                mn  = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
                mxb = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
                base = mw @ ((mn + mxb) * 0.5)
            if orientation == 'GLOBAL':
                direction = Vector((0.0, 0.0, 1.0))
            else:
                _, rq, _ = mw.decompose()
                direction = (rq.to_matrix() @ Vector((0.0, 0.0, 1.0))).normalized()
            bb  = [Vector(c) for c in obj.bound_box]
            bbm = max(
                max(v.x for v in bb) - min(v.x for v in bb),
                max(v.y for v in bb) - min(v.y for v in bb),
                max(v.z for v in bb) - min(v.z for v in bb), 0.01)
            _, _, sv = mw.decompose()
            ws = max(abs(sv.x), abs(sv.y), abs(sv.z), 1e-6)
            length = bbm * ws * scale_ratio
            tip = base + direction * length
            perp = direction.cross(Vector((0.0, 1.0, 0.0)))
            if perp.length < 0.001:
                perp = direction.cross(Vector((1.0, 0.0, 0.0)))
            perp  = perp.normalized()
            perp2 = direction.cross(perp).normalized()
            ch = length * 0.28; cr = length * 0.09
            cb = tip - direction * ch
            p1 = cb + perp*cr; p2 = cb + perp2*cr
            p3 = cb - perp*cr; p4 = cb - perp2*cr
            return [
                base[:], cb[:],
                tip[:], p1[:], tip[:], p2[:], tip[:], p3[:], tip[:], p4[:],
                p1[:], p2[:], p2[:], p3[:], p3[:], p4[:], p4[:], p1[:],
            ]
        except Exception as e:
            print(f"[Radix] arrow vert compute failed: {e}")
            return []

    def _arrow_color(obj):
        """Arrow colour: single mode → global black-default picker; MO mode →
        per-object override or the object's own quad colour."""
        return get_arrow_color(obj, scene)

    # Build colour → verts mapping
    color_buckets = {}   # {(r,g,b): [verts]}

    if cache_obj_name:
        try:
            ao = bpy.context.active_object
            if ao and ao.select_get() and ao.name == cache_obj_name and ao.type in GEOMETRY_TYPES:
                if not _persistent_previews.get(cache_obj_name, {}).get('muted', False):
                    verts = _arrow_verts_live(
                        ao,
                        _preview_cache.get('arrow_snap_local'),
                        _preview_cache.get('arrow_scale_ratio', 0.4),
                        _preview_cache.get('orientation', 'LOCAL'))
                    if verts:
                        c = _arrow_color(ao)
                        color_buckets.setdefault(c, []).extend(verts)
        except Exception as e:
            print(f"[Radix] active arrow failed: {e}")

    if scene.origin_keep_preview_persistent or draw_mo:
        active_n = getattr(bpy.context.active_object, 'name', None)
        arrow_sel_filter = None
        if draw_mo and not scene.origin_keep_preview_persistent:
            arrow_sel_filter = {o.name for o in bpy.context.selected_objects}
        for pname, pdata in list(_persistent_previews.items()):
            if pname == active_n or pdata.get('muted', False):
                continue
            if arrow_sel_filter is not None and pname not in arrow_sel_filter:
                continue
            po = bpy.data.objects.get(pname)
            if not po or po.type not in GEOMETRY_TYPES:
                continue
            verts = _arrow_verts_live(
                po,
                pdata.get('arrow_snap_local'),
                pdata.get('arrow_scale_ratio', 0.4),
                pdata.get('orientation', 'LOCAL'))
            if verts:
                c = _arrow_color(po)
                color_buckets.setdefault(c, []).extend(verts)

    gpu.state.depth_test_set('NONE')

    # Optional stroke (OFF by default for performance). Drawn as ONE batch for
    # all arrows — a single style (colour/width/opacity), not per-arrow casing —
    # so enabling it costs one extra draw call total, not one per object.
    if getattr(scene, 'origin_arrow_show_stroke', False) and color_buckets:
        try:
            stroke_verts = []
            for vlist in color_buckets.values():
                stroke_verts.extend(vlist)
            if stroke_verts:
                sc = tuple(scene.origin_arrow_stroke_color)
                sa = float(scene.origin_arrow_stroke_opacity)
                gpu.state.line_width_set(float(scene.origin_arrow_stroke_width))
                sbatch = batch_for_shader(shader, 'LINES', {"pos": stroke_verts})
                shader.bind()
                shader.uniform_float("color", (sc[0], sc[1], sc[2], sa))
                sbatch.draw(shader)
        except Exception as e:
            print(f"[Radix] arrow stroke draw failed: {e}")

    # Fill: one batch per colour bucket, single fixed width (no per-bucket
    # state churn).
    gpu.state.line_width_set(1.6)
    for arrow_color, arrow_verts in color_buckets.items():
        try:
            ab2 = batch_for_shader(shader, 'LINES', {"pos": arrow_verts})
            shader.bind()
            shader.uniform_float("color", (*arrow_color, 1.0))
            ab2.draw(shader)
        except Exception as e:
            print(f"[Radix] arrow draw failed: {e}")
    gpu.state.line_width_set(1.0)
    gpu.state.depth_test_set('LESS_EQUAL')

    # ── Viewport BBox Handles ─────────────────────────────────────────────────
    if getattr(scene, 'origin_show_viewport_handles', False):
        try:
            obj = bpy.context.active_object
            if obj and obj.type == 'MESH':
                _draw_viewport_handles(scene, obj)
        except Exception:
            pass

    gpu.state.depth_mask_set(True)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS_EQUAL')

def get_collection_color(obj):
    """
    Get color for object with 4-tier priority:
    1. Custom color (if set by user via RGB picker)
    2. Blender's collection tag color (from Properties Panel)
    3. Single-object color (if MO mode was just enabled)
    4. Random color (fallback)
    """
    global _persistent_previews

    scene = bpy.context.scene
    obj_name = obj.name

    # PRIORITY 1: Custom color
    if obj_name in _persistent_previews:
        custom = _persistent_previews[obj_name].get('custom_color')
        if custom is not None:
            return custom

    # PRIORITY 2: Blender collection tag (check cache first)
    if obj_name in _persistent_previews:
        cached_collection_color = _persistent_previews[obj_name].get('collection_color')
        if cached_collection_color is not None:
            return cached_collection_color

    # Detect collection color
    primary_collection = None
    for col in obj.users_collection:
        if col.name != "Scene Collection" and col != scene.collection:
            primary_collection = col
            break

    if primary_collection is None:
        primary_collection = scene.collection

    if hasattr(primary_collection, 'color_tag') and primary_collection.color_tag != 'NONE':
        blender_color = BLENDER_COLLECTION_COLORS.get(primary_collection.color_tag)
        if blender_color:
            if obj_name not in _persistent_previews:
                _persistent_previews[obj_name] = {}
            _persistent_previews[obj_name]['collection_color'] = blender_color
            return blender_color

    # PRIORITY 3: Fallback
    if obj_name in _persistent_previews:
        cached_fallback = _persistent_previews[obj_name].get('fallback_color')
        if cached_fallback is not None:
            return cached_fallback

    # Generate new fallback
    import random
    fallback = (
        random.uniform(0.3, 1.0),
        random.uniform(0.3, 1.0),
        random.uniform(0.3, 1.0)
    )

    if obj_name not in _persistent_previews:
        _persistent_previews[obj_name] = {}
    _persistent_previews[obj_name]['fallback_color'] = fallback

    return fallback


def mom_preview_active(scene=None):
    """Derived state — Multi-Object PREVIEW is active when the preview is ON and
    2+ geometry objects are selected.

    This is intentionally NOT a stored toggle. Computing it from the live
    selection means it can never be mutated from a depsgraph callback and can
    never fight the user. It is fully independent of 'Keep Persistent'.
    """
    if scene is None:
        scene = bpy.context.scene
    if not getattr(scene, 'show_origin_preview', False):
        return False
    try:
        sel = [o for o in bpy.context.selected_objects if o.type in GEOMETRY_TYPES]
        return len(sel) >= 2
    except Exception:
        return False


def get_arrow_color(obj, scene=None):
    """Resolve the arrow colour.

    Single mode → global ``origin_arrow_color`` (default black, user-pickable).
    Multi mode  → the object's own quad colour (collection/random) so the arrow
                  matches its highlight. No per-object arrow override.
    """
    if scene is None:
        scene = bpy.context.scene
    if mom_preview_active(scene):
        if obj and obj.type in GEOMETRY_TYPES:
            return tuple(get_collection_color(obj))
        return (0.0, 0.0, 0.0)
    return tuple(scene.origin_arrow_color)


# ---------------- Preview management (cache + handlers) ----------------
def draw_outliner_axis_indicators():
    """POST_PIXEL callback on the Outliner: draw a small coloured dot next to
    each object whose committed front-axis is non-default.

    The dot colour matches the axis: +Y (default) → grey (suppressed), -Y → cyan,
    +Z → green, -Z → magenta, +X → orange, -X → blue.  A grey dot would clutter
    the outliner, so objects using the Blender-default LOCAL_Y_POS get nothing.

    The callback is very cheap: it only fires when the Outliner is redrawn
    (user scrolling, scene change) and does a single font.draw() per marked object
    rather than geometry batches.
    """
    try:
        import blf
        ctx = bpy.context
        if ctx is None:
            return
        # Only run when objects carry the custom prop
        marked = {}
        for obj in bpy.data.objects:
            fa = obj.get('radixpro_front_axis', 'LOCAL_Y_POS')
            if fa and fa != 'LOCAL_Y_POS':
                marked[obj.name] = fa

        if not marked:
            return

        AXIS_COLORS = {
            'LOCAL_Y_NEG': (0.2, 0.9, 0.9, 1.0),   # cyan
            'LOCAL_Z_POS': (0.4, 0.9, 0.2, 1.0),   # green
            'LOCAL_Z_NEG': (0.9, 0.2, 0.8, 1.0),   # magenta
            'LOCAL_X_POS': (1.0, 0.55, 0.1, 1.0),  # orange
            'LOCAL_X_NEG': (0.2, 0.5,  1.0, 1.0),  # blue
        }
        AXIS_LABELS = {
            'LOCAL_Y_NEG': '−Y', 'LOCAL_Z_POS': '+Z', 'LOCAL_Z_NEG': '−Z',
            'LOCAL_X_POS': '+X', 'LOCAL_X_NEG': '−X',
        }

        region = ctx.region
        if not region or region.width < 10:
            return

        font_id = 0
        blf.size(font_id, 10)
        x_pos = region.width - 28  # right-aligned, inside the outliner column

        # Walk the visible tree — we can only approximate row positions without
        # direct outliner RNA access, so we draw a legend stripe at top-right instead,
        # listing non-default objects: name  axis-label  coloured dot.
        y = region.height - 22
        for name, fa in list(marked.items()):
            if y < 8:
                break
            color = AXIS_COLORS.get(fa, (0.8, 0.8, 0.8, 1.0))
            label = AXIS_LABELS.get(fa, fa[-1])
            blf.color(font_id, *color)
            blf.position(font_id, x_pos, y, 0)
            blf.draw(font_id, label)
            y -= 14
    except Exception:
        pass  # never crash the outliner


def ensure_handlers():
    global _draw_handler_3d, _draw_handler_2d, _draw_handler_outliner
    if _draw_handler_3d is None:
        _draw_handler_3d = bpy.types.SpaceView3D.draw_handler_add(
            draw_overlay_3d, (), 'WINDOW', 'POST_VIEW')
    if _draw_handler_2d is None:
        _draw_handler_2d = bpy.types.SpaceView3D.draw_handler_add(
            draw_handle_shapes_2d, (), 'WINDOW', 'POST_PIXEL')
    if _draw_handler_outliner is None:
        _draw_handler_outliner = bpy.types.SpaceOutliner.draw_handler_add(
            draw_outliner_axis_indicators, (), 'WINDOW', 'POST_PIXEL')


def remove_handlers():
    global _draw_handler_3d, _draw_handler_2d, _draw_handler_outliner
    if _draw_handler_3d is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler_3d, 'WINDOW')
        except Exception:
            pass
        _draw_handler_3d = None
    if _draw_handler_2d is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler_2d, 'WINDOW')
        except Exception:
            pass
        _draw_handler_2d = None
    if _draw_handler_outliner is not None:
        try:
            bpy.types.SpaceOutliner.draw_handler_remove(_draw_handler_outliner, 'WINDOW')
        except Exception:
            pass
        _draw_handler_outliner = None

def ensure_preview_collection():
    """Get or create the preview collection for arrow objects"""
    col = bpy.data.collections.get(PREVIEW_COLLECTION)
    if col is None:
        col = bpy.data.collections.new(PREVIEW_COLLECTION)
        try:
            bpy.context.scene.collection.children.link(col)
        except Exception:
            pass  # intentional: try:
    return col

def create_or_update_preview(obj):
    """Create or update preview plane (GPU) and arrow (mesh empty) to match active object."""
    global _preview_cache
    global _persistent_previews

    scene = bpy.context.scene
    source = scene.origin_snap_source
    obj_name = obj.name

    # Defer orphan cleanup — this function is reached from the depsgraph handler,
    # where synchronous object removal can recurse. The timer path is data-safe.
    schedule_cleanup()

    mo_active = mom_preview_active(scene)

    # Determine preview color
    if mo_active:
        # Multi-Object: per-object collection/random colour (priority-resolved)
        preview_color = get_collection_color(obj)

        # Ensure preview entry exists
        if obj_name not in _persistent_previews:
            _persistent_previews[obj_name] = {'muted': False}
        # Cache the active color for GPU draw
        _persistent_previews[obj_name]['active_color'] = preview_color
    else:
        # Single Object Mode: Use global color slider
        preview_color = scene.origin_preview_color
        # Keep any stored entry's cached colour in sync so quad and arrow never
        # diverge between modes.
        if obj_name in _persistent_previews:
            _persistent_previews[obj_name]['active_color'] = tuple(preview_color)

    # Determine arrow name (legacy; arrows are GPU-drawn, no real object)
    if mo_active or scene.origin_keep_preview_persistent:
        preview_arrow_name = f"{PREVIEW_ARROW_NAME}_{obj.name}"
    else:
        preview_arrow_name = PREVIEW_ARROW_NAME

    # Get vertices for preview
    if source == 'MESH':
        obj_eval, mesh = get_evaluated_mesh(obj)
        try:
            verts_local = [v.co.copy() for v in mesh.vertices]
            mw = obj_eval.matrix_world
        finally:
            release_evaluated_mesh(obj_eval, mesh)
    else:
        verts_local = get_bbox_local(obj)
        mw = obj.matrix_world

    if not verts_local:
        return

    bbox_verts_local = get_bbox_local(obj)

    min_x_local = min(bbox_verts_local, key=lambda v: v.x)
    max_x_local = max(bbox_verts_local, key=lambda v: v.x)
    min_y_local = min(bbox_verts_local, key=lambda v: v.y)
    max_y_local = max(bbox_verts_local, key=lambda v: v.y)
    min_z_local = min(bbox_verts_local, key=lambda v: v.z)
    max_z_local = max(bbox_verts_local, key=lambda v: v.z)

    xmid_local = (min_x_local.x + max_x_local.x) / 2.0
    ymid_local = (min_y_local.y + max_y_local.y) / 2.0
    zmid_local = (min_z_local.z + max_z_local.z) / 2.0
    bbox_center_world = obj.matrix_world @ Vector((xmid_local, ymid_local, zmid_local))

    orientation = scene.origin_preview_orientation

    # --- Resolve which local axis is "front" ---
    # get_front_axis_params works in LOCAL space; both GLOBAL and LOCAL modes
    # build the quad in local space first, then transform to world.
    # In GLOBAL mode the quad is built from world-transformed bbox verts so the
    # face aligns to world axes, but the front-axis choice still controls which
    # face of the *object* is highlighted.
    front_val_local, quad_fn_local, side_key_fn = get_front_axis_params(scene, bbox_verts_local)
    tol_local = max(
        (max_x_local.x - min_x_local.x),
        (max_y_local.y - min_y_local.y),
        (max_z_local.z - min_z_local.z),
        1e-6
    ) * 1e-3 + 1e-6

    # Pick which coordinate to filter on based on the active front axis
    fa = scene.origin_front_axis
    if fa in ('LOCAL_Y_POS', 'LOCAL_Y_NEG'):
        filter_key = lambda v: v.y
    elif fa in ('LOCAL_X_POS', 'LOCAL_X_NEG'):
        filter_key = lambda v: v.x
    else:  # Z axes
        filter_key = lambda v: v.z

    front_local_verts = [v for v in bbox_verts_local if abs(filter_key(v) - front_val_local) < tol_local]

    if front_local_verts:
        fx_min_local = min(side_key_fn(v) for v in front_local_verts)
        fx_max_local = max(side_key_fn(v) for v in front_local_verts)
        fz_min_local = min(v.z for v in front_local_verts) if fa not in ('LOCAL_Z_POS', 'LOCAL_Z_NEG') else min(v.y for v in front_local_verts)
        fz_max_local = max(v.z for v in front_local_verts) if fa not in ('LOCAL_Z_POS', 'LOCAL_Z_NEG') else max(v.y for v in front_local_verts)
    else:
        fx_min_local = min(side_key_fn(v) for v in bbox_verts_local)
        fx_max_local = max(side_key_fn(v) for v in bbox_verts_local)
        fz_min_local = min_z_local.z if fa not in ('LOCAL_Z_POS', 'LOCAL_Z_NEG') else min_y_local.y
        fz_max_local = max_z_local.z if fa not in ('LOCAL_Z_POS', 'LOCAL_Z_NEG') else max_y_local.y

    if orientation == 'GLOBAL':
        # GLOBAL: everything in world space. Front-face axis maps to world axes.
        # The quad is world-axis-aligned (sides parallel to world X/Z or X/Y etc.)
        bbox_verts_world = [obj.matrix_world @ v for v in bbox_verts_local]
        tol_world = 1e-5

        if fa == 'LOCAL_Y_POS':
            fv_world = max(v.y for v in bbox_verts_world)
            front_w  = [v for v in bbox_verts_world if abs(v.y - fv_world) < tol_world]
            if not front_w: front_w = bbox_verts_world
            fx_min = min(v.x for v in front_w); fx_max = max(v.x for v in front_w)
            fz_min = min(v.z for v in front_w); fz_max = max(v.z for v in front_w)
            quad_world = [Vector((fx_min, fv_world, fz_max)), Vector((fx_max, fv_world, fz_max)),
                          Vector((fx_max, fv_world, fz_min)), Vector((fx_min, fv_world, fz_min))]

        elif fa == 'LOCAL_Y_NEG':
            fv_world = min(v.y for v in bbox_verts_world)
            front_w  = [v for v in bbox_verts_world if abs(v.y - fv_world) < tol_world]
            if not front_w: front_w = bbox_verts_world
            fx_min = min(v.x for v in front_w); fx_max = max(v.x for v in front_w)
            fz_min = min(v.z for v in front_w); fz_max = max(v.z for v in front_w)
            # Reversed winding → outward normal = 0,-1,0
            quad_world = [Vector((fx_max, fv_world, fz_max)), Vector((fx_min, fv_world, fz_max)),
                          Vector((fx_min, fv_world, fz_min)), Vector((fx_max, fv_world, fz_min))]

        elif fa == 'LOCAL_X_POS':
            fv_world = max(v.x for v in bbox_verts_world)
            front_w  = [v for v in bbox_verts_world if abs(v.x - fv_world) < tol_world]
            if not front_w: front_w = bbox_verts_world
            fy_min = min(v.y for v in front_w); fy_max = max(v.y for v in front_w)
            fz_min = min(v.z for v in front_w); fz_max = max(v.z for v in front_w)
            # Winding → outward normal = +1,0,0
            quad_world = [Vector((fv_world, fy_max, fz_max)), Vector((fv_world, fy_min, fz_max)),
                          Vector((fv_world, fy_min, fz_min)), Vector((fv_world, fy_max, fz_min))]

        elif fa == 'LOCAL_X_NEG':
            fv_world = min(v.x for v in bbox_verts_world)
            front_w  = [v for v in bbox_verts_world if abs(v.x - fv_world) < tol_world]
            if not front_w: front_w = bbox_verts_world
            fy_min = min(v.y for v in front_w); fy_max = max(v.y for v in front_w)
            fz_min = min(v.z for v in front_w); fz_max = max(v.z for v in front_w)
            # Winding → outward normal = -1,0,0
            quad_world = [Vector((fv_world, fy_min, fz_max)), Vector((fv_world, fy_max, fz_max)),
                          Vector((fv_world, fy_max, fz_min)), Vector((fv_world, fy_min, fz_min))]

        elif fa == 'LOCAL_Z_POS':
            fv_world = max(v.z for v in bbox_verts_world)
            front_w  = [v for v in bbox_verts_world if abs(v.z - fv_world) < tol_world]
            if not front_w: front_w = bbox_verts_world
            fx_min = min(v.x for v in front_w); fx_max = max(v.x for v in front_w)
            fy_min = min(v.y for v in front_w); fy_max = max(v.y for v in front_w)
            # Winding → outward normal = 0,0,+1
            quad_world = [Vector((fx_min, fy_min, fv_world)), Vector((fx_max, fy_min, fv_world)),
                          Vector((fx_max, fy_max, fv_world)), Vector((fx_min, fy_max, fv_world))]

        else:  # LOCAL_Z_NEG
            fv_world = min(v.z for v in bbox_verts_world)
            front_w  = [v for v in bbox_verts_world if abs(v.z - fv_world) < tol_world]
            if not front_w: front_w = bbox_verts_world
            fx_min = min(v.x for v in front_w); fx_max = max(v.x for v in front_w)
            fy_min = min(v.y for v in front_w); fy_max = max(v.y for v in front_w)
            # Winding → outward normal = 0,0,-1
            quad_world = [Vector((fx_min, fy_max, fv_world)), Vector((fx_max, fy_max, fv_world)),
                          Vector((fx_max, fy_min, fv_world)), Vector((fx_min, fy_min, fv_world))]

    else:  # LOCAL — quad built in object local space, transformed to world
        quad_world = [
            obj.matrix_world @ v
            for v in quad_fn_local(fx_min_local, fx_max_local, fz_min_local, fz_max_local, front_val_local)
        ]

    # Convert quad_world back to local space for storage.
    # LOCAL quads: invert matrix_world. GLOBAL quads: store as-is in world space
    # but tag them so the draw function knows to use them directly (they ARE
    # world-space but will be recomputed at draw time via bbox_local + matrix).
    mw_inv = obj.matrix_world.inverted_safe()
    if orientation == 'LOCAL':
        quad_local_stored = [mw_inv @ v for v in quad_world]
    else:
        # For GLOBAL mode, store the local bbox verts — draw function rebuilds quad
        quad_local_stored = list(bbox_verts_local)  # 8 bbox verts in local space

    # ALWAYS update the main cache for current active object
    _preview_cache['quad_world']  = quad_world   # immediate draw (correct at snap time)
    _preview_cache['quad_local']  = quad_local_stored
    _preview_cache['obj_name']    = obj_name
    _preview_cache['orientation'] = orientation
    _preview_cache['front_axis']  = scene.origin_front_axis
    _preview_cache['bbox_local']  = list(bbox_verts_local)
    ensure_handlers()

    # Store preview data when persistent OR when in derived Multi-Object mode
    # (MO needs per-object quad/arrow data so the draw handler can render each).
    if scene.origin_keep_preview_persistent or mo_active:
        if obj_name not in _persistent_previews:
            _persistent_previews[obj_name] = {'color': preview_color, 'muted': False,
                                               '_obj_ref': obj}
        _persistent_previews[obj_name]['quad']        = quad_world
        _persistent_previews[obj_name]['quad_local']  = quad_local_stored
        _persistent_previews[obj_name]['orientation'] = orientation
        _persistent_previews[obj_name]['front_axis']  = scene.origin_front_axis
        _persistent_previews[obj_name]['bbox_local']  = list(bbox_verts_local)

    # ── Arrow data — stored for GPU draw, NO real object created ─────────────
    # Store LOCAL-space snap point so the draw handler can recompute world
    # position every frame via obj.matrix_world — arrow follows the object live.
    if obj_name in _last_snap_target:
        arrow_snap_local = _last_snap_target[obj_name].copy()
    else:
        arrow_snap_local = None   # draw handler falls back to live bbox centre

    # Arrow scale stored as a RATIO of bbox max dimension, not absolute units.
    # At draw time: length = bbox_world_max_dim * ratio. This keeps the arrow
    # visible and proportional regardless of object size.
    arrow_scale_ratio = max(scene.origin_arrow_scale, 0.01)

    _preview_cache['arrow_snap_local'] = arrow_snap_local
    _preview_cache['arrow_scale_ratio'] = arrow_scale_ratio

    # Muting
    is_muted = False
    if (scene.origin_keep_preview_persistent or mo_active) and obj_name in _persistent_previews:
        is_muted = _persistent_previews[obj_name].get('muted', False)

    if scene.origin_keep_preview_persistent or mo_active:
        if obj_name not in _persistent_previews:
            _persistent_previews[obj_name] = {'muted': False}
        _persistent_previews[obj_name]['arrow_snap_local']  = arrow_snap_local
        _persistent_previews[obj_name]['arrow_scale_ratio'] = arrow_scale_ratio
        _persistent_previews[obj_name]['muted']             = is_muted

    _tag_view3d_redraw()

# ---------------- Depsgraph Handler ----------------
def depsgraph_update_handler(scene, depsgraph):
    """Update preview dynamically when scene changes."""
    global _depsgraph_updating
    global _last_active_object_name
    global _preview_cache, _persistent_previews

    if _depsgraph_updating:
        return

    preview_on = getattr(scene, "show_origin_preview", False)

    # ── Preview turned OFF ────────────────────────────────────────────────────
    # The draw handler already early-exits, hiding the GPU quad. But arrow mesh
    # objects persist in bpy.data until we clean them here. Do it once, cheaply,
    # by checking whether any arrows still exist.
    if not preview_on:
        if _preview_cache or _persistent_previews:
            _preview_cache.clear()
            _persistent_previews.clear()
            schedule_cleanup()   # deferred bpy.data removal — safe from here
        return

    # ── Preview ON from here ──────────────────────────────────────────────────
    ensure_handlers()

    # NOTE: Multi-Object PREVIEW is now a DERIVED state (see mom_preview_active).
    # We deliberately do NOT mutate origin_multi_object_preview or
    # origin_keep_preview_persistent here. Writing scene properties from inside a
    # depsgraph callback re-enters the handler and fires their update callbacks,
    # which was the root cause of the preview/arrow/colour instability.

    obj = bpy.context.active_object

    current_name = obj.name if obj else None
    if current_name != _last_active_object_name:
        _last_active_object_name = current_name
        try:
            committed = get_obj_front_axis(obj, scene) if obj else 'LOCAL_Y_POS'
            if scene.origin_front_axis != committed:
                scene.origin_front_axis = committed
        except Exception:
            pass
        try:
            if obj and obj.type == 'MESH' and \
               getattr(scene, 'origin_offset_mode', 'OFFSET') == 'DIRECT':
                pos = obj.matrix_world.translation
                scene['_radixpro_suppress_offset_update'] = True
                scene.origin_offset_x = round(pos.x, 6)
                scene.origin_offset_y = round(pos.y, 6)
                scene.origin_offset_z = round(pos.z, 6)
                scene['_radixpro_suppress_offset_update'] = False
        except Exception:
            pass
        # Stamp cache immediately so draw handler shows the new object
        if obj and obj.type in GEOMETRY_TYPES:
            _preview_cache['obj_name']    = current_name
            _preview_cache['front_axis']  = get_obj_front_axis(obj, scene)
            _preview_cache['orientation'] = getattr(
                scene, 'origin_preview_orientation', 'LOCAL')

    # Auto-enable MO mode on shift-select is no longer needed — multi-object
    # preview is derived from the live selection (mom_preview_active).

    if not obj or obj.type not in GEOMETRY_TYPES:
        if not getattr(scene, "origin_keep_preview_persistent", False):
            remove_preview_objects()
        return

    if not getattr(scene, "origin_snap_auto_recalc", True):
        return

    _depsgraph_updating = True
    try:
        create_or_update_preview(obj)
        # Derived Multi-Object mode: build per-object preview data for the rest of
        # the selection. The draw handler recomputes world position from
        # matrix_world every frame, so transforms/camera moves need NO rebuild.
        # We therefore only (re)build an object when it has no cached local data
        # yet, or when its GEOMETRY actually changed this tick. This is what keeps
        # large multi-selections from rebuilding all N objects every frame.
        if mom_preview_active(scene):
            geo_changed = set()
            try:
                for upd in depsgraph.updates:
                    idd = getattr(upd, 'id', None)
                    if isinstance(idd, bpy.types.Object) and getattr(upd, 'is_updated_geometry', False):
                        geo_changed.add(idd.name)
            except Exception:
                pass
            for o in bpy.context.selected_objects:
                if o is obj or o.type not in GEOMETRY_TYPES:
                    continue
                cached = _persistent_previews.get(o.name)
                needs_build = (
                    cached is None
                    or 'quad_local' not in cached
                    or o.name in geo_changed
                )
                if needs_build:
                    try:
                        create_or_update_preview(o)
                    except Exception as e:
                        print(f"[Radix] multi preview build failed for '{o.name}': {e}")
    except Exception as e:
        print(f"[Radix] depsgraph update failed: {e}")
    finally:
        _depsgraph_updating = False

def collection_color_change_detector(scene, depsgraph):
    """
    Auto-sync when Blender's collection colors change in Properties Panel.
    Only updates objects that DON'T have custom color overrides.
    Fires whenever MO preview is active (derived state — no stored flag needed).
    """
    global _persistent_previews

    # Only run when multi-object preview is actually visible
    if not mom_preview_active(scene):
        return

    try:
        updated_count = 0

        for obj in bpy.data.objects:
            if obj.type != 'MESH':
                continue

            obj_name = obj.name

            # Skip if NOT in persistent previews
            if obj_name not in _persistent_previews:
                continue

            # Skip if object has CUSTOM color override
            custom = _persistent_previews[obj_name].get('custom_color')
            if custom is not None:
                continue

            # Get collection color
            primary_collection = None
            for col in obj.users_collection:
                if col.name != "Scene Collection" and col != scene.collection:
                    primary_collection = col
                    break

            if not primary_collection:
                continue

            if not hasattr(primary_collection, 'color_tag') or primary_collection.color_tag == 'NONE':
                continue

            # Get Blender's color
            blender_color = BLENDER_COLLECTION_COLORS.get(primary_collection.color_tag)
            if not blender_color:
                continue

            # Check if color changed
            cached_color = _persistent_previews[obj_name].get('collection_color')

            if cached_color != blender_color:
                # Update collection color
                _persistent_previews[obj_name]['collection_color'] = blender_color

                # Update preview if visible
                if scene.show_origin_preview:
                    try:
                        create_or_update_preview(obj)
                    except Exception:
                        pass  # intentional: try:

                updated_count += 1

        if updated_count > 0:
            _tag_view3d_redraw()

    except Exception:
        pass  # Silently fail

# ---------------- Callbacks for property updates ----------------
def keep_persistent_update(self, context):
    """Callback when Keep Persistent is toggled. Independent of Multi-Object mode."""
    scene = context.scene

    if not self.origin_keep_preview_persistent:
        # Toggled OFF - remove persisted (non-selected) previews
        remove_preview_objects()
    # Redraw either way
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

def origin_snap_source_update(self, context):
    """Update callback: if preview is on, refresh it when snap source changes."""
    scene = context.scene
    if scene.show_origin_preview:
        obj = context.active_object
        if obj and obj.type == 'MESH':
            try:
                create_or_update_preview(obj)
            except Exception:
                pass  # intentional: try:


def origin_arrow_scale_update(self, context):
    """Update callback: refresh preview when arrow scale changes."""
    scene = context.scene
    if scene.show_origin_preview:
        obj = context.active_object
        if obj and obj.type == 'MESH':
            try:
                create_or_update_preview(obj)
            except Exception:
                pass  # intentional: try:


def origin_flip_left_right_update(self, context):
    """Update callback: force UI redraw when flip toggle changes."""
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def origin_preview_orientation_update(self, context):
    """Update callback: refresh preview when orientation changes."""
    scene = context.scene
    if scene.show_origin_preview:
        obj = context.active_object
        if obj and obj.type == 'MESH':
            try:
                create_or_update_preview(obj)
            except Exception:
                pass  # intentional: try:

def origin_show_preview_controls_update(self, context):
    """Legacy no-op. Expand/collapse is now driven by show_origin_preview directly."""
    pass

def origin_multi_object_preview_update(self, context):
    """Legacy callback. Multi-Object preview is now derived from the live
    selection (mom_preview_active), so this no longer builds previews — it only
    requests a redraw in case the property is still set via the API."""
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

# ---------------- Multi-Object Target Selection ----------------
def get_affected_objects(context):
    """Return list of objects to be affected based on current settings.

    MMO auto-activates:
    - ALL_SELECTED: activates automatically when multiple meshes are selected —
      no preview toggle required.
    - CUSTOM_OBJECTS / COLLECTION: always uses those scopes regardless of
      selection count or preview state.
    Preview mode is no longer required to activate multi-object operation.
    """
    scene = context.scene
    affect_target = scene.origin_mo_affect_target

    if affect_target == 'CUSTOM_OBJECTS':
        result = []
        vl_objects = context.view_layer.objects
        for item in scene.origin_mo_custom_objects:
            obj = item.obj
            # PointerProperty can go stale after origin_set — fall back to name lookup
            if obj is None and item.name:
                obj = vl_objects.get(item.name)
            if obj and obj.name in vl_objects and obj.type == 'MESH' and obj.library is None:
                result.append(obj)
        # Fallback: custom list empty → use active object
        if not result:
            obj = context.active_object
            return [obj] if obj and obj.type == 'MESH' and obj.library is None else []
        return result

    elif affect_target == 'COLLECTION':
        col = scene.origin_mo_target_collection
        if not col:
            # No collection chosen → fallback to active object
            obj = context.active_object
            return [obj] if obj and obj.type == 'MESH' and obj.library is None else []
        return [obj for obj in col.objects
                if obj.type == 'MESH' and obj.library is None]

    else:  # ALL_SELECTED (default) — auto-MMO: all selected when multiple, active when single
        selected_meshes = [obj for obj in context.selected_objects
                           if obj.type == 'MESH' and obj.library is None]
        if selected_meshes:
            return selected_meshes
        # Fallback: active object only
        obj = context.active_object
        return [obj] if obj and obj.type == 'MESH' and obj.library is None else []
def auto_assign_multi_object_colors(context):
    """Auto-assign colors to all selected objects when MO mode is enabled"""
    global _persistent_previews

    scene = context.scene

    # Only run when conditions are met
    if not scene.origin_keep_preview_persistent:
        return

    if not scene.origin_multi_object_preview:
        return

    # Get all selected mesh objects
    selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']

    if len(selected_meshes) <= 1:
        return  # Only apply to multiple objects

    # Ensure all objects have colors assigned
    for obj in selected_meshes:
        obj_name = obj.name

        # Skip if already has a custom color
        if obj_name in _persistent_previews and _persistent_previews[obj_name].get('custom_color') is not None:
            continue

        # Use get_collection_color() to assign proper color
        color = get_collection_color(obj)

        # Create preview
        if scene.show_origin_preview:
            try:
                create_or_update_preview(obj)
            except Exception as e:
                print(f"[Origin Snap] Failed to create preview for {obj.name}: {e}")


# ---------------- Operators ----------------

def _calculate_snap_position(mode, obj, source, context):
    """Module-level snap position calculator — callable without instantiating
    the registered operator class (which bpy_struct forbids post-registration).

    Used by batch_normalize preview to compute target positions without moving
    any origins. Mirrors calculate_single_extreme exactly.
    """
    # Delegate through a temporary duck-typed namespace so the existing
    # method body runs unchanged with self.mode resolved.
    class _NS:
        pass
    ns = _NS()
    ns.mode = mode
    # Bind the three helper methods onto the namespace
    ns.get_vertex_ideal_local = lambda *a, **k: OBJECT_OT_set_origin_extreme_full.get_vertex_ideal_local(ns, *a, **k)
    ns.get_edge_ideal_local   = lambda *a, **k: OBJECT_OT_set_origin_extreme_full.get_edge_ideal_local(ns, *a, **k)
    ns.get_center_ideal_local = lambda *a, **k: OBJECT_OT_set_origin_extreme_full.get_center_ideal_local(ns, *a, **k)
    return OBJECT_OT_set_origin_extreme_full.calculate_single_extreme(ns, obj, source, context)


class OBJECT_OT_set_origin_extreme_full(bpy.types.Operator):
    """Set origin to geometry extremes (local-axis aware when Mesh mode selected)."""
    bl_idname = "object.set_origin_extreme_full"
    bl_label = "Set Origin to Geometry Extreme"
    bl_options = {'UNDO'}

    mode: bpy.props.StringProperty()

    def execute(self, context):
        scene = context.scene
        source = scene.origin_snap_source

        affected_objects = get_affected_objects(context)

        if not affected_objects:
            # Give a specific reason based on current mode
            target = scene.origin_mo_affect_target
            if target == 'ALL_SELECTED':
                self.report({'WARNING'}, "No mesh objects selected — select objects in the viewport first")
            elif target == 'CUSTOM_OBJECTS':
                self.report({'WARNING'}, "Custom object list is empty — add objects via Set Origin To panel")
            elif target == 'COLLECTION':
                if not scene.origin_mo_target_collection:
                    self.report({'WARNING'}, "No collection selected — pick a target collection in Set Origin To panel")
                else:
                    self.report({'WARNING'}, f"Collection '{scene.origin_mo_target_collection.name}' has no mesh objects")
            else:
                self.report({'WARNING'}, "No valid mesh objects to process")
            return {'CANCELLED'}

        # Special handling for CENTER_MASS - use Blender's native operator
        if self.mode == "CENTER_MASS":
            # Store original selection
            original_selection = [obj for obj in context.selected_objects]
            original_active = context.view_layer.objects.active

            for obj in affected_objects:
                # Deselect all first
                deselect_all_objects(context)

                # Select only this object
                obj.select_set(True)
                context.view_layer.objects.active = obj

                # Use Blender's native center of mass operator
                _safe_origin_set(context, type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')

                try:
                    push_snap_history(obj.name, obj.matrix_world.translation.copy())
                except Exception:
                    pass

            # Restore original selection
            deselect_all_objects(context)
            for obj in original_selection:
                if obj.name in context.view_layer.objects:
                    obj.select_set(True)
            if original_active and original_active.name in context.view_layer.objects:
                context.view_layer.objects.active = original_active
            context.view_layer.update()

            self.report({'INFO'}, f"✓ Center of Mass - {len(affected_objects)} object(s)")
            
            # Update preview if enabled
            if scene.show_origin_preview:
                obj = context.active_object
                if obj and obj.type == 'MESH':
                    try:
                        create_or_update_preview(obj)
                    except Exception:
                        pass  # intentional: try:
            
            return {'FINISHED'}

        snap_mode = scene.origin_mo_snap_mode if len(affected_objects) > 1 else 'INDIVIDUAL'

        if snap_mode == 'COMBINED' and len(affected_objects) > 1:
            # COMBINED MODE: Calculate ONE target for all objects
            target_world = self.calculate_combined_extreme(affected_objects, source, context)

            if target_world is None:
                self.report({'WARNING'}, "Could not calculate combined extreme")
                return {'CANCELLED'}

            # Store original selection
            original_selection = [obj for obj in context.selected_objects]
            original_active = context.view_layer.objects.active

            # Cursor snap: override combined target too
            if source == 'CURSOR':
                target_world = context.scene.cursor.location.copy()

            # Apply SAME target to ALL objects
            for obj in affected_objects:
                per_obj_target = _apply_lock_and_offset(obj, target_world, scene)
                _record_history(obj)

                # Deselect all first
                deselect_all_objects(context)

                # Select only this object
                obj.select_set(True)
                context.view_layer.objects.active = obj

                # Set cursor and apply origin
                context.scene.cursor.location = per_obj_target
                _safe_origin_set(context, type='ORIGIN_CURSOR', center='MEDIAN')

                # Record after origin_set: snap point is now at local (0,0,0)
                _last_snap_target[obj.name] = Vector((0, 0, 0))

                try:
                    push_snap_history(obj.name, per_obj_target.copy())
                except Exception:
                    pass

            # Restore original selection
            deselect_all_objects(context)
            for obj in original_selection:
                if obj.name in context.view_layer.objects:
                    obj.select_set(True)
            if original_active and original_active.name in context.view_layer.objects:
                context.view_layer.objects.active = original_active
            context.view_layer.update()

            # Stash post-snap base for live offset (before offset was applied)
            # Use the pre-offset target so dragging offsets from the snap point
            context.scene['_radixpro_offset_base'] = list(target_world - Vector((
                context.scene.origin_offset_x,
                context.scene.origin_offset_y,
                context.scene.origin_offset_z,
            )))

        else:
            # INDIVIDUAL MODE: Calculate SEPARATE target for EACH object
            # Store original selection
            original_selection = [obj for obj in context.selected_objects]
            original_active = context.view_layer.objects.active

            for obj in affected_objects:
                target_world = self.calculate_single_extreme(obj, source, context)

                if target_world is None:
                    continue

                # Cursor snap: override target with 3D cursor position
                if source == 'CURSOR':
                    target_world = context.scene.cursor.location.copy()

                # Apply axis lock + numeric offset
                target_world = _apply_lock_and_offset(obj, target_world, scene)

                # Record history before moving
                _record_history(obj)

                # Deselect all first
                deselect_all_objects(context)

                # Select only this object
                obj.select_set(True)
                context.view_layer.objects.active = obj

                # Set cursor and apply origin
                context.scene.cursor.location = target_world
                _safe_origin_set(context, type='ORIGIN_CURSOR', center='MEDIAN')

                # Record after origin_set: snap point is now at local (0,0,0)
                _last_snap_target[obj.name] = Vector((0, 0, 0))

                try:
                    push_snap_history(obj.name, target_world.copy())
                except Exception:
                    pass

            # Restore original selection
            deselect_all_objects(context)
            for obj in original_selection:
                if obj.name in context.view_layer.objects:
                    obj.select_set(True)
            if original_active and original_active.name in context.view_layer.objects:
                context.view_layer.objects.active = original_active
            context.view_layer.update()

            # Stash post-snap base for live offset on the active object
            active_after = context.active_object
            if active_after and active_after.type == 'MESH':
                pos = active_after.matrix_world.translation
                context.scene['_radixpro_offset_base'] = list(Vector((
                    pos.x - context.scene.origin_offset_x,
                    pos.y - context.scene.origin_offset_y,
                    pos.z - context.scene.origin_offset_z,
                )))
                # In Direct mode, sync fields to reflect the new landed position
                if getattr(context.scene, 'origin_offset_mode', 'OFFSET') == 'DIRECT':
                    context.scene['_radixpro_suppress_offset_update'] = True
                    context.scene.origin_offset_x = round(pos.x, 6)
                    context.scene.origin_offset_y = round(pos.y, 6)
                    context.scene.origin_offset_z = round(pos.z, 6)
                    context.scene['_radixpro_suppress_offset_update'] = False

        # ✅ Report operation success - AFTER all origin setting is complete
        if affected_objects:
            desc = self.get_mode_description(context)

            if snap_mode == 'COMBINED' and len(affected_objects) > 1:
                self.report({'INFO'}, f"✓ {desc} - Combined {len(affected_objects)} objects")
            else:
                if len(affected_objects) == 1:
                    obj = affected_objects[0]
                    target = self.calculate_single_extreme(obj, source, context)
                    if target:
                        self.report({'INFO'}, f"✓ {desc} at X:{target.x:.3f}, Y:{target.y:.3f}, Z:{target.z:.3f}")
                    else:
                        self.report({'INFO'}, f"✓ {desc}")
                else:
                    self.report({'INFO'}, f"✓ {desc} - {len(affected_objects)} objects (Independent)")

        if scene.show_origin_preview:
            obj = context.active_object
            if obj and obj.type == 'MESH':
                try:
                    create_or_update_preview(obj)
                except Exception:
                    pass  # intentional: try:

        # Store face normal for Normal Offset (face/center ops have well-defined normals)
        _FACE_NORMALS_LOCAL = {
            'FACE_TOP':    Vector(( 0,  0,  1)), 'CENTER_TOP':    Vector(( 0,  0,  1)),
            'FACE_BOTTOM': Vector(( 0,  0, -1)), 'CENTER_BOTTOM': Vector(( 0,  0, -1)),
            'FACE_FRONT':  Vector(( 0,  1,  0)), 'CENTER_FRONT':  Vector(( 0,  1,  0)),
            'FACE_BACK':   Vector(( 0, -1,  0)), 'CENTER_BACK':   Vector(( 0, -1,  0)),
            'FACE_LEFT':   Vector((-1,  0,  0)), 'CENTER_LEFT':   Vector((-1,  0,  0)),
            'FACE_RIGHT':  Vector(( 1,  0,  0)), 'CENTER_RIGHT':  Vector(( 1,  0,  0)),
        }
        if self.mode in _FACE_NORMALS_LOCAL:
            normal_local = _FACE_NORMALS_LOCAL[self.mode]
            for aobj in affected_objects:
                if scene.origin_preview_orientation == 'LOCAL':
                    nw = (aobj.matrix_world.to_3x3() @ normal_local).normalized()
                else:
                    nw = normal_local.normalized()
                _last_snap_normals[aobj.name] = nw

        return {'FINISHED'}

    def calculate_combined_extreme(self, objects, source, context):
        """Calculate combined extreme for multiple objects."""
        scene = context.scene

        is_face_operation = self.mode.startswith("FACE_")

        if is_face_operation:
            # Face operations use area-weighted centroid
            if self.mode == "FACE_TOP":
                return get_combined_face_centroid(objects, 'Z', 'max', source)
            elif self.mode == "FACE_BOTTOM":
                return get_combined_face_centroid(objects, 'Z', 'min', source)
            elif self.mode == "FACE_LEFT":
                extreme = 'min' if scene.origin_flip_left_right else 'max'
                return get_combined_face_centroid(objects, 'X', extreme, source)
            elif self.mode == "FACE_RIGHT":
                extreme = 'max' if scene.origin_flip_left_right else 'min'
                return get_combined_face_centroid(objects, 'X', extreme, source)
            elif self.mode == "FACE_FRONT":
                return get_combined_face_centroid(objects, 'Y', 'max', source)
            elif self.mode == "FACE_BACK":
                return get_combined_face_centroid(objects, 'Y', 'min', source)
        else:
            # For Edges, Centers, and Vertices: calculate based on source mode

            # STEP 1: Collect bbox corners (always needed for combined bounds)
            all_bbox_corners_world = []

            for obj in objects:
                if obj.type != 'MESH':
                    continue

                # Always get bbox for combined bounds calculation
                bbox_local = get_bbox_local(obj)
                bbox_world = [obj.matrix_world @ v for v in bbox_local]
                all_bbox_corners_world.extend(bbox_world)

            if not all_bbox_corners_world:
                return None

            # STEP 2: Calculate combined bounding box from bbox corners
            min_x_world = min(v.x for v in all_bbox_corners_world)
            max_x_world = max(v.x for v in all_bbox_corners_world)
            min_y_world = min(v.y for v in all_bbox_corners_world)
            max_y_world = max(v.y for v in all_bbox_corners_world)
            min_z_world = min(v.z for v in all_bbox_corners_world)
            max_z_world = max(v.z for v in all_bbox_corners_world)

            xmid_world = (min_x_world + max_x_world) / 2.0
            ymid_world = (min_y_world + max_y_world) / 2.0
            zmid_world = (min_z_world + max_z_world) / 2.0

            # STEP 3: Calculate ideal world position based on mode
            ideal_world = self.get_combined_ideal_world(
                scene,
                min_x_world, max_x_world,
                min_y_world, max_y_world,
                min_z_world, max_z_world,
                xmid_world, ymid_world, zmid_world
            )

            if ideal_world is None:
                return None

            # STEP 4: Return based on mode type and source
            if self.mode.startswith("VERT_"):
                # For VERTICES: snap to closest vertex from appropriate source
                if source == 'MESH':
                    # Collect mesh vertices ONLY when needed for VERTICES mode
                    all_verts_world = []
                    for obj in objects:
                        if obj.type != 'MESH':
                            continue
                        obj_eval, mesh = get_evaluated_mesh(obj)
                        try:
                            verts_world = [obj_eval.matrix_world @ v.co for v in mesh.vertices]
                            all_verts_world.extend(verts_world)
                        finally:
                            release_evaluated_mesh(obj_eval, mesh)

                    if not all_verts_world:
                        return None

                    # Snap to closest mesh vertex
                    best_vert = None
                    best_dist = None
                    for v in all_verts_world:
                        dist = (v - ideal_world).length
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_vert = v
                    return best_vert
                else:
                    # BBOX: Use the exact calculated corner position (not closest corner)
                    # For Combined mode, ideal_world is already the correct combined bbox corner
                    return ideal_world

            elif self.mode.startswith("EDGE_") or self.mode.startswith("CENTER_"):
                # For EDGES and CENTERS:
                if source == 'MESH':
                    # In MESH mode: snap to closest vertex across all objects
                    all_verts_world = []
                    for obj in objects:
                        if obj.type != 'MESH':
                            continue
                        obj_eval, mesh = get_evaluated_mesh(obj)
                        try:
                            verts_world = [obj_eval.matrix_world @ v.co for v in mesh.vertices]
                            all_verts_world.extend(verts_world)
                        finally:
                            release_evaluated_mesh(obj_eval, mesh)

                    if not all_verts_world:
                        return None

                    # Snap to closest mesh vertex
                    best_vert = None
                    best_dist = None
                    for v in all_verts_world:
                        dist = (v - ideal_world).length
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_vert = v
                    return best_vert
                else:
                    # In BBOX mode: return exact calculated position (no vertex snapping)
                    return ideal_world

            return None

    def get_combined_ideal_world(self, scene, min_x, max_x, min_y, max_y, min_z, max_z, xmid, ymid, zmid):
        """Get ideal world position for combined bounding box based on mode."""
        flip = scene.origin_flip_left_right

        # VERTICES
        if self.mode == 'VERT_TLF':
            return Vector((min_x if flip else max_x, max_y, max_z))
        elif self.mode == 'VERT_TRF':
            return Vector((max_x if flip else min_x, max_y, max_z))
        elif self.mode == 'VERT_TLB':
            return Vector((min_x if flip else max_x, min_y, max_z))
        elif self.mode == 'VERT_TRB':
            return Vector((max_x if flip else min_x, min_y, max_z))
        elif self.mode == 'VERT_BLF':
            return Vector((min_x if flip else max_x, max_y, min_z))
        elif self.mode == 'VERT_BRF':
            return Vector((max_x if flip else min_x, max_y, min_z))
        elif self.mode == 'VERT_BLB':
            return Vector((min_x if flip else max_x, min_y, min_z))
        elif self.mode == 'VERT_BRB':
            return Vector((max_x if flip else min_x, min_y, min_z))

        # EDGES
        elif self.mode == 'EDGE_TOP_LEFT':
            return Vector((min_x if flip else max_x, ymid, max_z))
        elif self.mode == 'EDGE_TOP_RIGHT':
            return Vector((max_x if flip else min_x, ymid, max_z))
        elif self.mode == 'EDGE_TOP_FRONT':
            return Vector((xmid, max_y, max_z))
        elif self.mode == 'EDGE_TOP_BACK':
            return Vector((xmid, min_y, max_z))
        elif self.mode == 'EDGE_BOT_LEFT':
            return Vector((min_x if flip else max_x, ymid, min_z))
        elif self.mode == 'EDGE_BOT_RIGHT':
            return Vector((max_x if flip else min_x, ymid, min_z))
        elif self.mode == 'EDGE_BOT_FRONT':
            return Vector((xmid, max_y, min_z))
        elif self.mode == 'EDGE_BOT_BACK':
            return Vector((xmid, min_y, min_z))
        elif self.mode == 'EDGE_FRONT_LEFT':
            return Vector((min_x if flip else max_x, max_y, zmid))
        elif self.mode == 'EDGE_FRONT_RIGHT':
            return Vector((max_x if flip else min_x, max_y, zmid))
        elif self.mode == 'EDGE_BACK_LEFT':
            return Vector((min_x if flip else max_x, min_y, zmid))
        elif self.mode == 'EDGE_BACK_RIGHT':
            return Vector((max_x if flip else min_x, min_y, zmid))

        # CENTERS
        elif self.mode == 'CENTER_TOP':
            return Vector((xmid, ymid, max_z))
        elif self.mode == 'CENTER_BOTTOM':
            return Vector((xmid, ymid, min_z))
        elif self.mode == 'CENTER_LEFT':
            return Vector((min_x if flip else max_x, ymid, zmid))
        elif self.mode == 'CENTER_RIGHT':
            return Vector((max_x if flip else min_x, ymid, zmid))
        elif self.mode == 'CENTER_FRONT':
            return Vector((xmid, max_y, zmid))
        elif self.mode == 'CENTER_BACK':
            return Vector((xmid, min_y, zmid))

        return None

    def calculate_single_extreme(self, obj, source, context):
        """Calculate extreme for a single object."""
        scene = context.scene

        if source == 'MESH':
            obj_eval, mesh = get_evaluated_mesh(obj)
            try:
                verts_local = [v.co.copy() for v in mesh.vertices]
                mw = obj_eval.matrix_world
            finally:
                release_evaluated_mesh(obj_eval, mesh)
        else:
            verts_local = get_bbox_local(obj)
            mw = obj.matrix_world

        if not verts_local:
            return None

        min_x_local = min(verts_local, key=lambda v: v.x)
        max_x_local = max(verts_local, key=lambda v: v.x)
        min_y_local = min(verts_local, key=lambda v: v.y)
        max_y_local = max(verts_local, key=lambda v: v.y)
        min_z_local = min(verts_local, key=lambda v: v.z)
        max_z_local = max(verts_local, key=lambda v: v.z)

        xmid_local = (min_x_local.x + max_x_local.x) / 2.0
        ymid_local = (min_y_local.y + max_y_local.y) / 2.0
        zmid_local = (min_z_local.z + max_z_local.z) / 2.0

        target_world = None

        # --- Faces ---
        if self.mode == "FACE_TOP":
            if source == 'MESH':
                t = face_centroid_extreme(obj, 'Z', 'max')
                target_world = t if t is not None else local_to_world(obj,
                                                                      Vector((xmid_local, ymid_local, max_z_local.z)))
            else:
                target_world = local_to_world(obj, Vector((xmid_local, ymid_local, max_z_local.z)))

        elif self.mode == "FACE_BOTTOM":
            if source == 'MESH':
                t = face_centroid_extreme(obj, 'Z', 'min')
                target_world = t if t is not None else local_to_world(obj,
                                                                      Vector((xmid_local, ymid_local, min_z_local.z)))
            else:
                target_world = local_to_world(obj, Vector((xmid_local, ymid_local, min_z_local.z)))

        elif self.mode == "FACE_LEFT":
            if scene.origin_flip_left_right:
                if source == 'MESH':
                    t = face_centroid_extreme(obj, 'X', 'min')
                    target_world = t if t is not None else local_to_world(obj, Vector(
                        (min_x_local.x, ymid_local, zmid_local)))
                else:
                    target_world = local_to_world(obj, Vector((min_x_local.x, ymid_local, zmid_local)))
            else:
                if source == 'MESH':
                    t = face_centroid_extreme(obj, 'X', 'max')
                    target_world = t if t is not None else local_to_world(obj, Vector(
                        (max_x_local.x, ymid_local, zmid_local)))
                else:
                    target_world = local_to_world(obj, Vector((max_x_local.x, ymid_local, zmid_local)))

        elif self.mode == "FACE_RIGHT":
            if scene.origin_flip_left_right:
                if source == 'MESH':
                    t = face_centroid_extreme(obj, 'X', 'max')
                    target_world = t if t is not None else local_to_world(obj, Vector(
                        (max_x_local.x, ymid_local, zmid_local)))
                else:
                    target_world = local_to_world(obj, Vector((max_x_local.x, ymid_local, zmid_local)))
            else:
                if source == 'MESH':
                    t = face_centroid_extreme(obj, 'X', 'min')
                    target_world = t if t is not None else local_to_world(obj, Vector(
                        (min_x_local.x, ymid_local, zmid_local)))  # Use min_x for fallback
                else:
                    target_world = local_to_world(obj, Vector((min_x_local.x, ymid_local, zmid_local)))  # Use min_x for bbox

        elif self.mode == "FACE_FRONT":
            if source == 'MESH':
                t = face_centroid_extreme(obj, 'Y', 'max')
                target_world = t if t is not None else local_to_world(obj,
                                                                      Vector((xmid_local, max_y_local.y, zmid_local)))
            else:
                target_world = local_to_world(obj, Vector((xmid_local, max_y_local.y, zmid_local)))

        elif self.mode == "FACE_BACK":
            if source == 'MESH':
                t = face_centroid_extreme(obj, 'Y', 'min')
                target_world = t if t is not None else local_to_world(obj,
                                                                      Vector((xmid_local, min_y_local.y, zmid_local)))
            else:
                target_world = local_to_world(obj, Vector((xmid_local, min_y_local.y, zmid_local)))

        # --- Vertices ---
        elif self.mode.startswith("VERT_"):
            ideal_local = self.get_vertex_ideal_local(scene, min_x_local, max_x_local, min_y_local, max_y_local,
                                                        min_z_local, max_z_local)
            if source == 'MESH':
                obj_eval, mesh = get_evaluated_mesh(obj)
                try:
                    verts_local_eval = [v.co.copy() for v in mesh.vertices]
                    verts_world_eval = [obj_eval.matrix_world @ v.co for v in mesh.vertices]
                    idx = find_closest_vertex_index_local(verts_local_eval, ideal_local)
                    target_world = verts_world_eval[idx] if idx is not None else local_to_world(obj, ideal_local)
                finally:
                    release_evaluated_mesh(obj_eval, mesh)
            else:
                target_world = local_to_world(obj, ideal_local)

                # --- Edges ---
        elif self.mode.startswith("EDGE_"):
            ideal_local = self.get_edge_ideal_local(scene, min_x_local, max_x_local, min_y_local, max_y_local,
                                                        min_z_local, max_z_local, xmid_local, ymid_local, zmid_local)
            if source == 'MESH':
                # In Mesh mode: find TRUE edge midpoint by averaging vertices at both extremes
                flip = scene.origin_flip_left_right
                
                # Map edge mode to axis extremes
                edge_params = {
                    # Top ring edges (Z=max)
                    'EDGE_TOP_LEFT':    ('Z', 'max', 'X', 'min' if flip else 'max'),
                    'EDGE_TOP_RIGHT':   ('Z', 'max', 'X', 'max' if flip else 'min'),
                    'EDGE_TOP_FRONT':   ('Z', 'max', 'Y', 'max'),
                    'EDGE_TOP_BACK':    ('Z', 'max', 'Y', 'min'),
                    # Bottom ring edges (Z=min)
                    'EDGE_BOT_LEFT':    ('Z', 'min', 'X', 'min' if flip else 'max'),
                    'EDGE_BOT_RIGHT':   ('Z', 'min', 'X', 'max' if flip else 'min'),
                    'EDGE_BOT_FRONT':   ('Z', 'min', 'Y', 'max'),
                    'EDGE_BOT_BACK':    ('Z', 'min', 'Y', 'min'),
                    # Vertical edges (X and Y extremes)
                    'EDGE_FRONT_LEFT':  ('Y', 'max', 'X', 'min' if flip else 'max'),
                    'EDGE_FRONT_RIGHT': ('Y', 'max', 'X', 'max' if flip else 'min'),
                    'EDGE_BACK_LEFT':   ('Y', 'min', 'X', 'min' if flip else 'max'),
                    'EDGE_BACK_RIGHT':  ('Y', 'min', 'X', 'max' if flip else 'min'),
                }
                
                if self.mode in edge_params:
                    axis1, extreme1, axis2, extreme2 = edge_params[self.mode]
                    t = edge_midpoint_extreme(obj, axis1, extreme1, axis2, extreme2)
                    target_world = t if t is not None else local_to_world(obj, ideal_local)
                else:
                    target_world = local_to_world(obj, ideal_local)
            else:
                # In BBox mode: use exact calculated midpoint position
                target_world = local_to_world(obj, ideal_local)

            # --- Centers ---
        elif self.mode.startswith("CENTER_") and self.mode not in {"CENTER_GEOMETRY", "CENTER_BBOX", "CENTER_MASS"}:
            ideal_local = self.get_center_ideal_local(scene, min_x_local, max_x_local, min_y_local, max_y_local,
                                                      min_z_local, max_z_local, xmid_local, ymid_local, zmid_local)
            if source == 'MESH':
                # In Mesh mode: snap to closest vertex to the ideal face center
                obj_eval, mesh = get_evaluated_mesh(obj)
                try:
                    verts_local_eval = [v.co.copy() for v in mesh.vertices]
                    verts_world_eval = [obj_eval.matrix_world @ v.co for v in mesh.vertices]
                    idx = find_closest_vertex_index_local(verts_local_eval, ideal_local)
                    target_world = verts_world_eval[idx] if idx is not None else local_to_world(obj, ideal_local)
                finally:
                    release_evaluated_mesh(obj_eval, mesh)
            else:
                # In BBox mode: use exact calculated face center position
                target_world = local_to_world(obj, ideal_local)

        # --- Global Centers ---
        elif self.mode == "CENTER_GEOMETRY":
            # Center of Geometry (median point - average of all vertices)
            if source == 'MESH':
                obj_eval, mesh = get_evaluated_mesh(obj)
                try:
                    verts_local = [v.co.copy() for v in mesh.vertices]
                    if verts_local:
                        center_local = sum(verts_local, Vector()) / len(verts_local)
                        target_world = obj_eval.matrix_world @ center_local
                    else:
                        # Fallback to bbox center if no vertices
                        target_world = local_to_world(obj, Vector((xmid_local, ymid_local, zmid_local)))
                finally:
                    release_evaluated_mesh(obj_eval, mesh)
            else:
                # BBox mode: use bbox center
                target_world = local_to_world(obj, Vector((xmid_local, ymid_local, zmid_local)))

        elif self.mode == "CENTER_BBOX":
            # Center of BBox (geometric center of bounds - works in both modes)
            target_world = local_to_world(obj, Vector((xmid_local, ymid_local, zmid_local)))

        elif self.mode == "CENTER_MASS":
            # Center of Mass - use Blender's native calculation
            # We'll handle this specially in execute() using bpy.ops.object.origin_set
            # For now, return bbox center as fallback
            target_world = local_to_world(obj, Vector((xmid_local, ymid_local, zmid_local)))

        return target_world

    def get_vertex_ideal_local(self, scene, min_x, max_x, min_y, max_y, min_z, max_z):
        """Get ideal local position for vertex corners with flip support."""
        flip = scene.origin_flip_left_right

        corner_map = {
            'VERT_TLF': Vector((min_x.x if flip else max_x.x, max_y.y, max_z.z)),
            'VERT_TRF': Vector((max_x.x if flip else min_x.x, max_y.y, max_z.z)),
            'VERT_TLB': Vector((min_x.x if flip else max_x.x, min_y.y, max_z.z)),
            'VERT_TRB': Vector((max_x.x if flip else min_x.x, min_y.y, max_z.z)),
            'VERT_BLF': Vector((min_x.x if flip else max_x.x, max_y.y, min_z.z)),
            'VERT_BRF': Vector((max_x.x if flip else min_x.x, max_y.y, min_z.z)),
            'VERT_BLB': Vector((min_x.x if flip else max_x.x, min_y.y, min_z.z)),
            'VERT_BRB': Vector((max_x.x if flip else min_x.x, min_y.y, min_z.z)),
        }
        return corner_map.get(self.mode, Vector((0, 0, 0)))

    def get_edge_ideal_local(self, scene, min_x, max_x, min_y, max_y, min_z, max_z, xmid, ymid, zmid):
        """Get ideal local position for edge midpoints with flip support."""
        flip = scene.origin_flip_left_right

        edge_map = {
            'EDGE_TOP_LEFT': Vector((min_x.x if flip else max_x.x, ymid, max_z.z)),
            'EDGE_TOP_RIGHT': Vector((max_x.x if flip else min_x.x, ymid, max_z.z)),
            'EDGE_TOP_FRONT': Vector((xmid, max_y.y, max_z.z)),
            'EDGE_TOP_BACK': Vector((xmid, min_y.y, max_z.z)),
            'EDGE_BOT_LEFT': Vector((min_x.x if flip else max_x.x, ymid, min_z.z)),
            'EDGE_BOT_RIGHT': Vector((max_x.x if flip else min_x.x, ymid, min_z.z)),
            'EDGE_BOT_FRONT': Vector((xmid, max_y.y, min_z.z)),
            'EDGE_BOT_BACK': Vector((xmid, min_y.y, min_z.z)),
            'EDGE_FRONT_LEFT': Vector((min_x.x if flip else max_x.x, max_y.y, zmid)),
            'EDGE_FRONT_RIGHT': Vector((max_x.x if flip else min_x.x, max_y.y, zmid)),
            'EDGE_BACK_LEFT': Vector((min_x.x if flip else max_x.x, min_y.y, zmid)),
            'EDGE_BACK_RIGHT': Vector((max_x.x if flip else min_x.x, min_y.y, zmid)),
        }
        return edge_map.get(self.mode, Vector((xmid, ymid, zmid)))

    def get_center_ideal_local(self, scene, min_x, max_x, min_y, max_y, min_z, max_z, xmid, ymid, zmid):
        """Get ideal local position for face centers with flip support."""
        flip = scene.origin_flip_left_right

        center_map = {
            'CENTER_TOP': Vector((xmid, ymid, max_z.z)),
            'CENTER_BOTTOM': Vector((xmid, ymid, min_z.z)),
            'CENTER_LEFT': Vector((min_x.x if flip else max_x.x, ymid, zmid)),
            'CENTER_RIGHT': Vector((max_x.x if flip else min_x.x, ymid, zmid)),
            'CENTER_FRONT': Vector((xmid, max_y.y, zmid)),
            'CENTER_BACK': Vector((xmid, min_y.y, zmid)),
        }
        return center_map.get(self.mode, Vector((xmid, ymid, zmid)))

    def get_mode_description(self, context):
        """Get human-readable description of the mode."""
        scene = context.scene
        flip = scene.origin_flip_left_right

        descriptions = {
            'FACE_TOP': "Top Face (+Z)",
            'FACE_BOTTOM': "Bottom Face (-Z)",
            'FACE_LEFT': f"Left Face ({'-X' if flip else '+X'})",
            'FACE_RIGHT': f"Right Face ({'+X' if flip else '-X'})",
            'FACE_FRONT': "Front Face (+Y)",
            'FACE_BACK': "Back Face (-Y)",
        }

        return descriptions.get(self.mode, f"Mode: {self.mode}")

# ---------------- Supporting Operators ----------------


# ---------------- Surface Snap Operator (Free Surface Mode) ----------------


# ---------------- Vertex Snap Operator (Drag-to-Snap Mode) ----------------


# ---------------- Quick Snap Operators for Alt+Click ----------------


# ── Pie Menu ──────────────────────────────────────────────────────────────────

class VIEW3D_MT_radix_pie(bpy.types.Menu):
    """Radix Free quick-access pie menu (Alt+Q in 3D View)"""
    bl_label = "Radix Free"

    def draw(self, context):
        layout = self.layout
        pie    = layout.menu_pie()
        scene  = context.scene

        # W  — Left
        pie.operator("object.set_origin_extreme_full",
                     text="Left (+X)", icon='TRIA_LEFT').mode = 'FACE_LEFT'
        # E  — Right
        pie.operator("object.set_origin_extreme_full",
                     text="Right (-X)", icon='TRIA_RIGHT').mode = 'FACE_RIGHT'
        # S  — Bottom
        pie.operator("object.set_origin_extreme_full",
                     text="Bottom (-Z)", icon='TRIA_DOWN').mode = 'FACE_BOTTOM'
        # N  — Top
        pie.operator("object.set_origin_extreme_full",
                     text="Top (+Z)", icon='TRIA_UP').mode = 'FACE_TOP'
        # NW — Back
        pie.operator("object.set_origin_extreme_full",
                     text="Back (-Y)", icon='BACK').mode = 'FACE_BACK'
        # NE — Front
        pie.operator("object.set_origin_extreme_full",
                     text="Front (+Y)", icon='FORWARD').mode = 'FACE_FRONT'
        # SW — Center Geometry
        pie.operator("object.set_origin_extreme_full",
                     text="Center Geo", icon='PIVOT_MEDIAN').mode = 'CENTER_GEOMETRY'
        # SE — World Zero
        pie.operator("radixpro.snap_to_world_zero",
                     text="World Zero", icon='WORLD')


# ── Snap to World Grid ────────────────────────────────────────────────────────


# ── Normal Offset (apply along last stored snap normal) ───────────────────────


# ── Symmetry Origin ───────────────────────────────────────────────────────────


# ── Viewport BBox Handles — Click-to-Snap Modal ────────────────────────────────

class RADIX_OT_click_bbox_handle(bpy.types.Operator):
    """Click a viewport bounding-box handle to snap the origin there.
    Handles are drawn when 'Viewport Handles' is enabled in Origin Snap Mode."""
    bl_idname  = "radixpro.click_bbox_handle"
    bl_label   = "Click BBox Handle"
    bl_options = {'REGISTER', 'UNDO'}

    _draw_handler   = None
    _handles        = []        # [(world_pos, label, color)]
    _hovered_handle = None

    @classmethod
    def poll(cls, context):
        return (context.area.type == 'VIEW_3D' and
                context.active_object and
                context.active_object.type == 'MESH')

    def invoke(self, context, event):
        obj = context.active_object
        self._handles = self._build_handles(obj, context.scene)
        if not self._handles:
            self.report({'WARNING'}, "Cannot compute handles — object has no geometry")
            return {'CANCELLED'}

        self._hovered_handle = None
        self._draw_handler   = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_cb, (context,), 'WINDOW', 'POST_PIXEL'
        )
        context.window.cursor_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "Click a handle to snap origin  |  RMB / Esc: Cancel"
        )
        context.area.header_text_set(
            "BBox Handle Snap — click a handle dot  |  Esc: Cancel"
        )
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        context.area.tag_redraw()

        if event.type == 'MOUSEMOVE':
            self._hovered_handle = self._nearest_handle(context, event)
            # Share with the passive draw_handle_shapes_2d callback so it can
            # enlarge the hovered shape without needing a reference to this instance.
            _handle_hover_world_pos[0] = (
                self._hovered_handle[0].copy() if self._hovered_handle else None
            )
            return {'RUNNING_MODAL'}

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            handle = self._nearest_handle(context, event)
            if handle:
                self._snap_to(context, handle[0])
                self._cleanup(context)
                self.report({'INFO'}, f"✓ Origin snapped to {handle[1]}")
                return {'FINISHED'}
            # Click missed all handles — cancel
            self._cleanup(context)
            return {'CANCELLED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cleanup(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_handles(self, obj, scene):
        """Build handle list: (world_pos, label, rgb_color)."""
        mw    = obj.matrix_world
        bbl   = [Vector(c) for c in obj.bound_box]
        bbw   = [mw @ v for v in bbl]

        xs = [v.x for v in bbw]; ys = [v.y for v in bbw]; zs = [v.z for v in bbw]
        mn = Vector((min(xs), min(ys), min(zs)))
        mx = Vector((max(xs), max(ys), max(zs)))
        md = (mn + mx) / 2

        mode   = scene.origin_handles_snap_type
        result = []

        if mode in ('FACE_CENTERS', 'ALL'):
            # 6 face centers — blue
            result += [
                (Vector((md.x, md.y, mx.z)), "Top Center",    (0.3, 0.6, 1.0)),
                (Vector((md.x, md.y, mn.z)), "Bottom Center", (0.3, 0.6, 1.0)),
                (Vector((md.x, mx.y, md.z)), "Front Center",  (0.3, 0.6, 1.0)),
                (Vector((md.x, mn.y, md.z)), "Back Center",   (0.3, 0.6, 1.0)),
                (Vector((mx.x, md.y, md.z)), "Right Center",  (0.3, 0.6, 1.0)),
                (Vector((mn.x, md.y, md.z)), "Left Center",   (0.3, 0.6, 1.0)),
            ]

        if mode in ('CORNERS', 'ALL'):
            # 8 corners — orange
            for v in bbw:
                result.append((v.copy(), "Corner", (1.0, 0.55, 0.1)))

        if mode in ('EDGE_MIDPOINTS', 'ALL'):
            # 12 edge midpoints — green
            edges = [
                (0,1),(1,2),(2,3),(3,0),    # bottom ring
                (4,5),(5,6),(6,7),(7,4),    # top ring
                (0,4),(1,5),(2,6),(3,7),    # verticals
            ]
            for a, b in edges:
                mid = (bbw[a] + bbw[b]) / 2
                result.append((mid, "Edge Midpoint", (0.2, 0.9, 0.4)))

        if mode == 'BBOX_CENTER':
            result.append((md.copy(), "BBox Center", (1.0, 0.9, 0.1)))

        return result

    def _nearest_handle(self, context, event, threshold_px=16):
        """Return the closest handle within threshold_px, or None."""
        region = context.region
        rv3d   = context.region_data
        if not region or not rv3d:
            return None

        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        best, best_d = None, float('inf')

        for handle in self._handles:
            screen = location_3d_to_region_2d(region, rv3d, handle[0])
            if screen:
                d = (Vector(screen) - mouse).length
                if d < threshold_px and d < best_d:
                    best_d = d
                    best   = handle

        return best

    def _snap_to(self, context, world_pos):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return
        _record_history(obj)
        orig_cursor = context.scene.cursor.location.copy()
        orig_sel    = list(context.selected_objects)
        orig_active = context.view_layer.objects.active

        context.scene.cursor.location = world_pos
        deselect_all_objects(context)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        _safe_origin_set(context, type='ORIGIN_CURSOR')

        try:
            push_snap_history(obj.name, Vector(world_pos))
        except Exception:
            pass

        deselect_all_objects(context)
        for o in orig_sel:
            if o.name in context.view_layer.objects:
                o.select_set(True)
        context.view_layer.objects.active = orig_active
        context.scene.cursor.location = orig_cursor

        if context.scene.show_origin_preview:
            try:
                create_or_update_preview(obj)
            except Exception:
                pass

    def _draw_cb(self, context):
        """Draw handle dots in POST_PIXEL space for sharp screen-space dots."""
        region = context.region
        rv3d   = context.region_data
        if not region or not rv3d:
            return

        gpu.state.blend_set('ALPHA')
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        for handle in self._handles:
            world_pos = handle[0]
            color     = handle[2]
            is_hov    = (self._hovered_handle is handle)
            size      = 14.0 if is_hov else 9.0
            alpha     = 1.0  if is_hov else 0.8
            sc        = location_3d_to_region_2d(region, rv3d, world_pos)
            if sc:
                try:
                    batch = batch_for_shader(shader, 'POINTS', {"pos": [(*sc, 0)]})
                    gpu.state.point_size_set(size)
                    shader.bind()
                    shader.uniform_float("color", (*color, alpha))
                    batch.draw(shader)
                except Exception:
                    pass

        gpu.state.point_size_set(1.0)
        gpu.state.blend_set('NONE')

    def _cleanup(self, context):
        _handle_hover_world_pos[0] = None   # drop hover so shapes return to normal size
        if self._draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        context.window.cursor_set('DEFAULT')
        context.area.header_text_set(None)
        context.workspace.status_text_set(None)
        context.area.tag_redraw()


# ---------------- Panel ----------------


# ═══════════════════════════════════════════════════════════════════════════════
# NEW v1.2.0 OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Origin History ────────────────────────────────────────────────────────────


# ── Copy / Paste Origin ───────────────────────────────────────────────────────


# ── Edit Mode Origin Snap ─────────────────────────────────────────────────────


# ── Batch Normalize ───────────────────────────────────────────────────────────


# ── Batch Normalize helpers (module-level so Blender never validates them) ─────

def _bn_update_preview(op, context):
    """Recompute world positions for the current snap_position selection."""
    op._last_snap = op.snap_position
    objects = [o for o in context.selected_objects
               if o.type == 'MESH' and o.library is None]
    op._preview_points = []
    for obj in objects:
        try:
            pt = _calculate_snap_position(
                op.snap_position, obj,
                context.scene.origin_snap_source, context)
            if pt is not None:
                op._preview_points.append(Vector(pt))
        except Exception:
            pass


def _bn_commit(op, context):
    """Apply the batch normalize to all selected objects."""
    objects = [o for o in context.selected_objects
               if o.type == 'MESH' and o.library is None]
    scene   = context.scene
    blop    = bpy.ops.object.set_origin_extreme_full

    saved_sel    = list(context.selected_objects)
    saved_active = context.view_layer.objects.active
    saved_cursor = context.scene.cursor.location.copy()
    count = 0

    for obj in objects:
        deselect_all_objects(context)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        _record_history(obj)
        result = blop(mode=op.snap_position)
        if 'FINISHED' in result:
            count += 1

    context.scene.cursor.location = saved_cursor
    deselect_all_objects(context)
    for o in saved_sel:
        if o and o.name in context.view_layer.objects:
            o.select_set(True)
    if saved_active and saved_active.name in context.view_layer.objects:
        context.view_layer.objects.active = saved_active

    if scene.show_origin_preview:
        active = context.active_object
        if active and active.type == 'MESH':
            try:
                create_or_update_preview(active)
            except Exception:
                pass

    op.report({'INFO'}, f"Batch normalized {count} object(s) to {op.snap_position}")
    return {'FINISHED'}


def _bn_draw_preview(op):
    """Draw green circle rings at each preview origin position (screen-space)."""
    if not getattr(op, '_preview_points', None):
        return
    try:
        import gpu, math
        from gpu_extras.batch import batch_for_shader
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        from mathutils import Matrix

        region = bpy.context.region
        rv3d   = bpy.context.space_data.region_3d if bpy.context.space_data else None
        if not region or not rv3d:
            return

        SEGMENTS   = 32
        RADIUS     = 14.0
        THICKNESS  = 2.5
        ARM        = RADIUS * 0.5

        COLOR_RING = (0.2, 0.95, 0.3, 1.0)
        COLOR_FILL = (0.2, 0.95, 0.3, 0.15)
        COLOR_CROSS= (1.0, 1.0, 1.0, 0.90)

        angles    = [2 * math.pi * i / SEGMENTS for i in range(SEGMENTS + 1)]
        unit_ring = [(math.cos(a), math.sin(a)) for a in angles]
        unit_fan  = []
        for i in range(SEGMENTS):
            unit_fan += [(0.0, 0.0),
                         (math.cos(angles[i]),   math.sin(angles[i])),
                         (math.cos(angles[i+1]), math.sin(angles[i+1]))]

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('ALPHA')

        w, h = region.width, region.height
        ortho = Matrix((
            (2/w,   0,  0, -1),
            (  0, 2/h,  0, -1),
            (  0,   0,  1,  0),
            (  0,   0,  0,  1),
        ))

        gpu.matrix.push()
        gpu.matrix.load_identity()
        gpu.matrix.load_projection_matrix(ortho)

        shader.bind()

        for pt3d in op._preview_points:
            co2d = location_3d_to_region_2d(region, rv3d, pt3d)
            if co2d is None:
                continue
            cx, cy = co2d

            fill = [(cx + x*RADIUS, cy + y*RADIUS) for x,y in unit_fan]
            shader.uniform_float("color", COLOR_FILL)
            batch_for_shader(shader,'TRIS',{"pos":fill}).draw(shader)

            ring = [(cx + x*RADIUS, cy + y*RADIUS) for x,y in unit_ring]
            gpu.state.line_width_set(THICKNESS)
            shader.uniform_float("color", COLOR_RING)
            batch_for_shader(shader,'LINE_STRIP',{"pos":ring}).draw(shader)

            cross = [(cx-ARM,cy),(cx+ARM,cy),(cx,cy-ARM),(cx,cy+ARM)]
            gpu.state.line_width_set(1.5)
            shader.uniform_float("color", COLOR_CROSS)
            batch_for_shader(shader,'LINES',{"pos":cross}).draw(shader)

        gpu.matrix.pop()
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('LESS_EQUAL')
    except Exception:
        pass


def _bn_cleanup(op):
    """Remove the preview draw handler."""
    h = getattr(op, '_draw_handler', None)
    if h is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
        except Exception:
            pass
        op._draw_handler = None
    _tag_view3d_redraw()



def _place_origin_any(o, world_pos, context):
    """Move an object's origin to world_pos, dispatching by type:
      MESH                         → geometry-preserving fast path (no undo spam)
      other geometry (curve/text…) → operator origin_set (Blender supports these)
      non-geometry (empty/light…)  → move the object itself (origin == object)
    """
    if o.type == 'MESH':
        _set_origin_world_fast(o, world_pos, context)
    elif o.type in GEOMETRY_TYPES:
        _apply_origin(o, world_pos, context)
    else:
        m = o.matrix_world.copy()
        m.translation = Vector(world_pos)
        o.matrix_world = m


# ═══════════════════════════════════════════════════════════════════════════════
# RADIX PRO — NEW FEATURE OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════


# ── FEATURE 3: Object-space offset toggle ─────────────────────────────────────
# Handled via scene.origin_offset_space prop + modified _apply_lock_and_offset
# (wired below in property registration)

# ── FEATURE 4: Snap to another object's origin (eyedropper picker) ────────────


# ── FEATURE 4b: Move object to another object's origin ───────────────────────


# ── FEATURE 5: Align origins across selection ─────────────────────────────────


# ── FEATURE 1: Snap to Curve (first/last/all points) ─────────────────────────


# ── FEATURE 2: Multi-object paste chain ───────────────────────────────────────


# ── FEATURE 6: Named persistent MO groups ────────────────────────────────────
    # Derived: list of names is stored as CSV string for persistence


# ── FEATURE 7: Surface snap to other scene objects ────────────────────────────

# ── FEATURE 9: CSV export of origin positions ─────────────────────────────────


# ── Presets ───────────────────────────────────────────────────────────────────


class VIEW3D_PT_origin_tools(bpy.types.Panel):
    """Radix — Precision Origin Placement (Free Edition)
39 snap positions via dropdown menus · BBox Handle mode"""
    bl_label = "Radix Free"
    bl_idname = "VIEW3D_PT_origin_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Radix Free'

    def draw(self, context):
        layout = self.layout
        scene  = context.scene
        obj    = context.active_object

        # ============ SNAP REFERENCE (Always visible) ============
        # Controls that affect how the 39 positions are calculated:
        # which geometry to read from, world vs local orientation, and
        # which local axis counts as "front" for Front/Back/Left/Right.
        box = layout.box()
        hdr = box.row(align=True)
        hdr.label(text="Snap Reference", icon='SNAP_ON')
        hdr.operator("wm.call_menu", text="", icon='MESH_ICOSPHERE',
                     emboss=False).name = 'VIEW3D_MT_radix_pie'
        hdr.operator("radixpro.show_shortcuts_info", text="", icon='QUESTION', emboss=False)

        box.prop(scene, "origin_snap_source", text="")
        box.prop(scene, "origin_preview_orientation", text="Orientation")

        row = box.row(align=True)
        row.label(text="Front Face Axis:", icon='ORIENTATION_LOCAL')
        row.prop(scene, "origin_front_axis", text="")

        layout.separator()

        # ============ SET ORIGIN TO — dropdown menus ============
        # All 39 snap positions, grouped exactly like Faces / Vertices /
        # Edge Midpoints / Centers. Each opens as a native Blender dropdown.
        if obj and obj.type == 'MESH':
            box = layout.box()
            box.label(text="Set Origin To", icon='OBJECT_ORIGIN')

            col = box.column(align=True)
            col.scale_y = 1.15
            col.menu("VIEW3D_MT_set_origin_faces",    icon='FACESEL')
            col.menu("VIEW3D_MT_set_origin_vertices", icon='VERTEXSEL')
            col.menu("VIEW3D_MT_set_origin_edges",    icon='EDGESEL')
            col.menu("VIEW3D_MT_set_origin_centers",  icon='PIVOT_BOUNDBOX')

            box.separator(factor=0.5)
            box.label(text="Global Centers:", icon='WORLD')
            grow = box.row(align=True)
            grow.operator("object.set_origin_extreme_full",
                         text="Geometry").mode = 'CENTER_GEOMETRY'
            grow.operator("object.set_origin_extreme_full",
                         text="BBox").mode = 'CENTER_BBOX'
            grow.operator("object.set_origin_extreme_full",
                         text="Mass").mode = 'CENTER_MASS'

            box.separator(factor=0.5)
            qrow = box.row(align=True)
            qrow.scale_y = 1.1
            qrow.operator("object.set_origin_extreme_full",
                         text="3D Cursor", icon='CURSOR').mode = 'CURSOR'
            qrow.operator("radixpro.snap_to_world_zero",
                         text="World Zero", icon='WORLD')
        else:
            layout.label(text="Select a mesh object", icon='ERROR')

        layout.separator()

        # ============ BBOX HANDLE MODE ============
        box = layout.box()
        box.label(text="BBox Handle Mode", icon='SNAP_ON')
        h_row = box.row(align=True)
        h_icon = 'HIDE_OFF' if scene.origin_show_viewport_handles else 'HIDE_ON'
        h_row.prop(scene, "origin_show_viewport_handles", text="Handles", icon=h_icon, toggle=True)
        if scene.origin_show_viewport_handles:
            h_row.prop(scene, "origin_handles_snap_type", text="")
            hc_row = box.row(align=True)
            hc_row.scale_y = 1.1
            hc_row.operator("radixpro.click_bbox_handle", text="Click Handle to Snap", icon='SNAP_ON')

        # ============ FOOTER ============
        layout.separator(factor=1.2)
        footer = layout.row(align=True)
        footer.scale_y = 1.3
        footer.operator("radixpro.open_basic_link", text="❤ Get Radix Basic", icon='NONE')
        footer.operator("radixpro.open_pro_link", text="❤ Get Radix Pro", icon='NONE')

# ---------------- Menus ----------------
def menu_draw_handler(self, context):
    """No-op in Radix Free — preview auto-show is a Basic/Pro feature."""
    pass

class VIEW3D_MT_set_origin_faces(bpy.types.Menu):
    bl_label = "Faces (Geometry)"

    def draw(self, context):
        menu_draw_handler(self, context)
        layout = self.layout
        scene = context.scene

        layout.operator("object.set_origin_extreme_full", text="Top (+Z)").mode = 'FACE_TOP'
        layout.operator("object.set_origin_extreme_full", text="Bottom (-Z)").mode = 'FACE_BOTTOM'

        if scene.origin_flip_left_right:
            layout.operator("object.set_origin_extreme_full", text="Left (+X)").mode = 'FACE_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Right (-X)").mode = 'FACE_RIGHT'
        else:
            layout.operator("object.set_origin_extreme_full", text="Left (-X)").mode = 'FACE_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Right (+X)").mode = 'FACE_RIGHT'

        layout.operator("object.set_origin_extreme_full", text="Front (+Y)").mode = 'FACE_FRONT'
        layout.operator("object.set_origin_extreme_full", text="Back (-Y)").mode = 'FACE_BACK'


class VIEW3D_MT_set_origin_vertices(bpy.types.Menu):
    bl_label = "Vertices (Corners)"

    def draw(self, context):
        menu_draw_handler(self, context)
        layout = self.layout
        scene = context.scene

        if scene.origin_flip_left_right:
            layout.operator("object.set_origin_extreme_full", text="Top Left Front (+X+Y+Z)").mode = 'VERT_TLF'
            layout.operator("object.set_origin_extreme_full", text="Top Right Front (-X+Y+Z)").mode = 'VERT_TRF'
            layout.operator("object.set_origin_extreme_full", text="Top Left Back (+X-Y+Z)").mode = 'VERT_TLB'
            layout.operator("object.set_origin_extreme_full", text="Top Right Back (-X-Y+Z)").mode = 'VERT_TRB'
            layout.separator()
            layout.operator("object.set_origin_extreme_full", text="Bottom Left Front (+X+Y-Z)").mode = 'VERT_BLF'
            layout.operator("object.set_origin_extreme_full", text="Bottom Right Front (-X+Y-Z)").mode = 'VERT_BRF'
            layout.operator("object.set_origin_extreme_full", text="Bottom Left Back (+X-Y-Z)").mode = 'VERT_BLB'
            layout.operator("object.set_origin_extreme_full", text="Bottom Right Back (-X-Y-Z)").mode = 'VERT_BRB'
        else:
            layout.operator("object.set_origin_extreme_full", text="Top Left Front (-X+Y+Z)").mode = 'VERT_TLF'
            layout.operator("object.set_origin_extreme_full", text="Top Right Front (+X+Y+Z)").mode = 'VERT_TRF'
            layout.operator("object.set_origin_extreme_full", text="Top Left Back (-X-Y+Z)").mode = 'VERT_TLB'
            layout.operator("object.set_origin_extreme_full", text="Top Right Back (+X-Y+Z)").mode = 'VERT_TRB'
            layout.separator()
            layout.operator("object.set_origin_extreme_full", text="Bottom Left Front (-X+Y-Z)").mode = 'VERT_BLF'
            layout.operator("object.set_origin_extreme_full", text="Bottom Right Front (+X+Y-Z)").mode = 'VERT_BRF'
            layout.operator("object.set_origin_extreme_full", text="Bottom Left Back (-X-Y-Z)").mode = 'VERT_BLB'
            layout.operator("object.set_origin_extreme_full", text="Bottom Right Back (+X-Y-Z)").mode = 'VERT_BRB'


class VIEW3D_MT_set_origin_edges(bpy.types.Menu):
    bl_label = "Edge Midpoints (12)"

    def draw(self, context):
        menu_draw_handler(self, context)
        layout = self.layout
        scene = context.scene

        layout.label(text="Top ring (+Z)")
        if scene.origin_flip_left_right:
            layout.operator("object.set_origin_extreme_full", text="Top Left (+X)").mode = 'EDGE_TOP_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Top Right (-X)").mode = 'EDGE_TOP_RIGHT'
        else:
            layout.operator("object.set_origin_extreme_full", text="Top Left (-X)").mode = 'EDGE_TOP_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Top Right (+X)").mode = 'EDGE_TOP_RIGHT'

        layout.operator("object.set_origin_extreme_full", text="Top Front (+Y)").mode = 'EDGE_TOP_FRONT'
        layout.operator("object.set_origin_extreme_full", text="Top Back (-Y)").mode = 'EDGE_TOP_BACK'
        layout.separator()

        layout.label(text="Bottom ring (-Z)")
        if scene.origin_flip_left_right:
            layout.operator("object.set_origin_extreme_full", text="Bottom Left (+X)").mode = 'EDGE_BOT_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Bottom Right (-X)").mode = 'EDGE_BOT_RIGHT'
        else:
            layout.operator("object.set_origin_extreme_full", text="Bottom Left (-X)").mode = 'EDGE_BOT_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Bottom Right (+X)").mode = 'EDGE_BOT_RIGHT'

        layout.operator("object.set_origin_extreme_full", text="Bottom Front (+Y)").mode = 'EDGE_BOT_FRONT'
        layout.operator("object.set_origin_extreme_full", text="Bottom Back (-Y)").mode = 'EDGE_BOT_BACK'
        layout.separator()

        layout.label(text="Middle edges")
        if scene.origin_flip_left_right:
            layout.operator("object.set_origin_extreme_full", text="Front Left (+X+Y)").mode = 'EDGE_FRONT_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Front Right (-X+Y)").mode = 'EDGE_FRONT_RIGHT'
            layout.operator("object.set_origin_extreme_full", text="Back Left (+X-Y)").mode = 'EDGE_BACK_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Back Right (-X-Y)").mode = 'EDGE_BACK_RIGHT'
        else:
            layout.operator("object.set_origin_extreme_full", text="Front Left (-X+Y)").mode = 'EDGE_FRONT_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Front Right (+X+Y)").mode = 'EDGE_FRONT_RIGHT'
            layout.operator("object.set_origin_extreme_full", text="Back Left (-X-Y)").mode = 'EDGE_BACK_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Back Right (+X-Y)").mode = 'EDGE_BACK_RIGHT'


class VIEW3D_MT_set_origin_centers(bpy.types.Menu):
    bl_label = "Centers (Face Centers)"

    def draw(self, context):
        menu_draw_handler(self, context)
        layout = self.layout
        scene = context.scene

        layout.operator("object.set_origin_extreme_full", text="Top (+Z)").mode = 'CENTER_TOP'
        layout.operator("object.set_origin_extreme_full", text="Bottom (-Z)").mode = 'CENTER_BOTTOM'

        if scene.origin_flip_left_right:
            layout.operator("object.set_origin_extreme_full", text="Left (+X)").mode = 'CENTER_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Right (-X)").mode = 'CENTER_RIGHT'
        else:
            layout.operator("object.set_origin_extreme_full", text="Left (-X)").mode = 'CENTER_LEFT'
            layout.operator("object.set_origin_extreme_full", text="Right (+X)").mode = 'CENTER_RIGHT'

        layout.operator("object.set_origin_extreme_full", text="Front (+Y)").mode = 'CENTER_FRONT'
        layout.operator("object.set_origin_extreme_full", text="Back (-Y)").mode = 'CENTER_BACK'


class VIEW3D_MT_set_origin_main(bpy.types.Menu):
    bl_label = "Set Origin to..."

    def draw(self, context):
        layout = self.layout

        # Snap to 3D Cursor and World Zero — quick deterministic targets
        layout.operator("object.set_origin_extreme_full",
                        text="Snap to 3D Cursor", icon='CURSOR').mode = 'CURSOR'
        layout.operator("radixpro.snap_to_world_zero",
                        text="Snap to World Zero", icon='WORLD')
        layout.separator()

        layout.menu("VIEW3D_MT_set_origin_faces")
        layout.menu("VIEW3D_MT_set_origin_vertices")
        layout.menu("VIEW3D_MT_set_origin_edges")
        layout.menu("VIEW3D_MT_set_origin_centers")
        layout.separator()

        layout.label(text="Global Centers")
        layout.operator("object.set_origin_extreme_full",
                        text="Center Geometry").mode = 'CENTER_GEOMETRY'
        layout.operator("object.set_origin_extreme_full",
                        text="Center BBox").mode = 'CENTER_BBOX'
        layout.operator("object.set_origin_extreme_full",
                        text="Center Mass").mode = 'CENTER_MASS'


# ---------------- Property Group for Custom Objects ----------------


# ---------------- Registration ----------------




# ══════════════════════════════════════════════════════════════════════════════
# RADIX PLACE — Phases 1, 2 & 3
# ══════════════════════════════════════════════════════════════════════════════

# ── Snap History helpers ───────────────────────────────────────────────────────

def push_snap_history(obj_name: str, world_pos: Vector) -> None:
    """Record a snap position. Called automatically by _apply_origin() and confirm_snap.
    Suppressed during history navigation and batch operations via _SNAP_RECORDING[0]."""
    if not _SNAP_RECORDING[0]:
        return
    entry = (obj_name, world_pos.copy())
    if _SNAP_HISTORY and _SNAP_HISTORY[-1] == entry:
        return
    _SNAP_HISTORY.append(entry)
    if len(_SNAP_HISTORY) > _SNAP_HISTORY_MAX:
        _SNAP_HISTORY.pop(0)
    _SNAP_HISTORY_IDX[0] = len(_SNAP_HISTORY) - 1


def clear_snap_history() -> None:
    _SNAP_HISTORY.clear()
    _SNAP_HISTORY_IDX[0] = 0


# ── Collision Preview helpers ──────────────────────────────────────────────────

def _rp_world_bbox(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    mx = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return mn, mx


def _rp_bboxes_overlap(a_min, a_max, b_min, b_max, tol=0.001):
    return (a_min.x <= b_max.x + tol and a_max.x >= b_min.x - tol and
            a_min.y <= b_max.y + tol and a_max.y >= b_min.y - tol and
            a_min.z <= b_max.z + tol and a_max.z >= b_min.z - tol)


def _draw_collision_overlay():
    """POST_VIEW: green wireframe = clear, red = bbox overlap."""
    try:
        ctx      = bpy.context
        selected = [o for o in ctx.selected_objects if o.type in GEOMETRY_TYPES]
        if len(selected) < 2:
            return
        bboxes = [(obj, *_rp_world_bbox(obj)) for obj in selected]
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(1.5)
        for i, (obj_a, a_min, a_max) in enumerate(bboxes):
            colliding = any(
                _rp_bboxes_overlap(a_min, a_max, b_min, b_max)
                for j, (obj_b, b_min, b_max) in enumerate(bboxes) if i != j
            )
            color = (0.9, 0.15, 0.1, 0.35) if colliding else (0.1, 0.9, 0.2, 0.25)
            mn, mx = a_min, a_max
            corners = [
                Vector((mn.x, mn.y, mn.z)), Vector((mx.x, mn.y, mn.z)),
                Vector((mx.x, mx.y, mn.z)), Vector((mn.x, mx.y, mn.z)),
                Vector((mn.x, mn.y, mx.z)), Vector((mx.x, mn.y, mx.z)),
                Vector((mx.x, mx.y, mx.z)), Vector((mn.x, mx.y, mx.z)),
            ]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            flat  = [p for a, b in edges for p in (tuple(corners[a]), tuple(corners[b]))]
            batch = batch_for_shader(shader, 'LINES', {"pos": flat})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('LESS_EQUAL')
    except Exception:
        pass


# ── Surface Align helper ───────────────────────────────────────────────────────

def _rp_align_axis_to_normal(obj, local_axis: str, world_normal: Vector) -> None:
    """Rotate obj so its chosen local axis aligns with world_normal."""
    sign = -1.0 if local_axis.startswith('-') else 1.0
    idx  = {'X': 0, 'Y': 1, 'Z': 2}[local_axis[-1]]
    target = world_normal.normalized() * sign
    _, rot_q, sca = obj.matrix_world.decompose()
    current_axis  = rot_q.to_matrix().col[idx].normalized()
    cross = current_axis.cross(target)
    if cross.length < 1e-6:
        return
    dot   = max(-1.0, min(1.0, current_axis.dot(target)))
    angle = math.acos(dot)
    rot   = Matrix.Rotation(angle, 4, cross.normalized())
    loc   = obj.matrix_world.translation.copy()
    new_rot = (rot.to_3x3() @ rot_q.to_matrix()).to_4x4()
    obj.matrix_world = Matrix.Translation(loc) @ new_rot @ Matrix.Diagonal((*sca, 1.0))


# ══ Phase 1 Operators ═════════════════════════════════════════════════════════


# ══ Phase 2 Operators ═════════════════════════════════════════════════════════


# ══ Radix Place Panel ═════════════════════════════════════════════════════════




# ══ F3: Snap History HUD draw function ════════════════════════════════════════

def _draw_history_hud():
    """POST_PIXEL: show a small history-position banner for ~2 s after Prev/Next."""
    import time
    try:
        if not _history_hud_text[0]:
            return
        elapsed = time.monotonic() - _history_hud_time[0]
        if elapsed > 2.5:
            _history_hud_text[0] = None
            return
        alpha = max(0.0, 1.0 - (elapsed - 1.5))   # fade last second
        import blf
        ctx    = bpy.context
        region = getattr(ctx, 'region', None)
        if not region:
            return
        blf.size(0, 18)
        blf.color(0, 0.95, 0.72, 0.18, alpha)
        blf.position(0, 18, region.height - 46, 0)
        blf.draw(0, _history_hud_text[0])
    except Exception:
        pass


# ══ F6: Viewport Mode Indicator draw function ══════════════════════════════════

def _draw_mode_indicator():
    """POST_PIXEL: small corner overlay showing current snap mode & orientation."""
    try:
        ctx   = bpy.context
        scene = ctx.scene
        if not getattr(scene, 'origin_show_mode_indicator', False):
            return
        region = getattr(ctx, 'region', None)
        if not region:
            return
        src   = getattr(scene, 'origin_snap_source',         'MESH')
        ori   = getattr(scene, 'origin_preview_orientation', 'LOCAL')
        lx    = getattr(scene, 'origin_lock_x', False)
        ly    = getattr(scene, 'origin_lock_y', False)
        lz    = getattr(scene, 'origin_lock_z', False)
        locks = ''.join([('X' if lx else ''), ('Y' if ly else ''), ('Z' if lz else '')])
        lock_str = f"  Lock:{locks}" if locks else ""
        import blf
        blf.size(0, 13)
        blf.color(0, 0.85, 0.85, 0.85, 0.65)
        blf.position(0, 12, 12 + 32, 0)
        blf.draw(0, f"Radix  Source:{src}  Orient:{ori}{lock_str}")
    except Exception:
        pass


# ══ F1: Snap to World Zero ════════════════════════════════════════════════════

class RADIX_OT_snap_to_world_zero(bpy.types.Operator):
    """Move this object's origin to (0, 0, 0) — the scene / world centre."""
    bl_idname  = "radixpro.snap_to_world_zero"
    bl_label   = "Origin to World Zero"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object and
                context.active_object.type in GEOMETRY_TYPES and
                not context.active_object.library)

    def execute(self, context):
        obj = context.active_object
        _record_history(obj)
        _apply_origin(obj, Vector((0.0, 0.0, 0.0)), context)
        self.report({'INFO'}, f"✓ Origin → World Zero (0,0,0)  [{obj.name}]")
        _tag_view3d_redraw()
        return {'FINISHED'}


def _bn_commit(op, context):
    """Apply the batch normalize to all selected objects."""
    objects = [o for o in context.selected_objects
               if o.type == 'MESH' and o.library is None]
    scene   = context.scene
    blop    = bpy.ops.object.set_origin_extreme_full

    saved_sel    = list(context.selected_objects)
    saved_active = context.view_layer.objects.active
    saved_cursor = context.scene.cursor.location.copy()
    count = 0

    for obj in objects:
        deselect_all_objects(context)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        _record_history(obj)
        result = blop(mode=op.snap_position)
        if 'FINISHED' in result:
            count += 1

    context.scene.cursor.location = saved_cursor
    deselect_all_objects(context)
    for o in saved_sel:
        if o and o.name in context.view_layer.objects:
            o.select_set(True)
    if saved_active and saved_active.name in context.view_layer.objects:
        context.view_layer.objects.active = saved_active

    if scene.show_origin_preview:
        active = context.active_object
        if active and active.type == 'MESH':
            try:
                create_or_update_preview(active)
            except Exception:
                pass

    op.report({'INFO'}, f"Batch normalized {count} object(s) to {op.snap_position}")
    return {'FINISHED'}


class RADIX_OT_open_basic_link(bpy.types.Operator):
    """Radix Basic adds:

• Surface / Vertex / Grid / Cursor click-snapping
• Viewport preview — highlight quad + direction arrow
• Snap History with Prev / Next navigation
• Multi-Object mode with per-object/collection colour
• Offset & Placement tools, Copy/Paste origin
• Symmetry Origin, Batch Normalize, Normal Offset
• Edit Mode Snap

Click to open Discord and get Radix Basic"""
    bl_idname  = "radixpro.open_basic_link"
    bl_label   = "Get Radix Basic"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        bpy.ops.wm.url_open(url="https://discord.com/users/damagedarchitect")
        return {'FINISHED'}


class RADIX_OT_open_pro_link(bpy.types.Operator):
    """Radix Pro adds everything in Basic, plus:

• Surface Snap modifiers — Ctrl=Grid  Alt=NormalOffset  X/Y/Z=AxisLock
• Numerical offset input while snapping
• Object Snap Tools — mesh-to-mesh snap, rotation alignment, group placement
• Group center & proportional copy/paste, Curve Point Snap
• Snap to Scene Surface (raycast), Multi-axis Align Origins
• Chain Paste, CSV export, Saved Configurations, Named Object Groups
• Snap History HUD, Viewport Mode Indicator
• Radix Place suite — Pivot Library, Collision Preview, Surface Alignment,
  Smart + Batch Contact Detection, Snap Layers, Chain / Distribute origins

Click to open Discord and get Radix Pro"""
    bl_idname  = "radixpro.open_pro_link"
    bl_label   = "Get Radix Pro"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        bpy.ops.wm.url_open(url="https://discord.com/users/damagedarchitect")
        return {'FINISHED'}


class RADIX_OT_show_shortcuts_info(bpy.types.Operator):
    """Radix keyboard shortcuts:

Alt + Q  →  Radix pie menu (8 quick snap positions)

Click-to-snap, Surface/Vertex snapping, and more shortcuts
are available in Radix Basic and Radix Pro."""
    bl_idname  = "radixpro.show_shortcuts_info"
    bl_label   = "Keyboard Shortcuts"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        return {'FINISHED'}


classes = (
    RADIX_OT_open_basic_link,
    RADIX_OT_open_pro_link,
    RADIX_OT_show_shortcuts_info,
    OBJECT_OT_set_origin_extreme_full,
    VIEW3D_MT_radix_pie,
    RADIX_OT_click_bbox_handle,
    VIEW3D_PT_origin_tools,
    VIEW3D_MT_set_origin_faces,
    VIEW3D_MT_set_origin_vertices,
    VIEW3D_MT_set_origin_edges,
    VIEW3D_MT_set_origin_centers,
    VIEW3D_MT_set_origin_main,
    # ── Radix Place Phase 1 ───────────────────────────────────────────────────
    # ── Radix Place Phase 2 ───────────────────────────────────────────────────
    # ── Radix Place Phase 3 ───────────────────────────────────────────────────
    # ── New operators (v3.1.0) ────────────────────────────────────────────────
    RADIX_OT_snap_to_world_zero,
)

def menu_func(self, context):
    self.layout.menu("VIEW3D_MT_set_origin_main")


def context_menu_func(self, context):
    self.layout.menu("VIEW3D_MT_set_origin_main")


# Keymap storage
addon_keymaps = []


def _on_load_post(*args):
    """Reset all volatile in-memory state when a new .blend file is loaded.

    Without this, _persistent_previews and _preview_cache can contain stale
    references to objects from the previous file — RNA pointers that are now
    invalid and will hard-crash Blender on next access.

    This handler is intentionally minimal: we clear caches and remove GPU
    handlers, then let the depsgraph handler re-register them naturally on the
    first viewport redraw in the new file.
    """
    global _persistent_previews, _preview_cache, _last_snap_target
    global _last_active_object_name, _cleanup_scheduled, _auto_show_timer

    _persistent_previews.clear()
    _preview_cache.clear()
    _last_snap_target.clear()
    _last_snap_normals.clear()
    _origin_clipboard.clear()
    _last_active_object_name = None
    _cleanup_scheduled = False
    _auto_show_timer = None

    # Clear Radix Place transient state
    clear_snap_history()
    _handle_hover_world_pos[0] = None
    if _collision_active[0]:
        if _collision_handler[0] is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(_collision_handler[0], 'WINDOW')
            except Exception:
                pass
        _collision_handler[0] = None
        _collision_active[0]  = False

    # Remove GPU draw handlers — they hold closures that reference old data.
    # ensure_handlers() will re-add the 3D ones on the next depsgraph tick when
    # the new file's preview activates. The outliner indicator is safe to
    # re-add immediately (it only reads custom props, no scene-data access).
    remove_handlers()
    global _draw_handler_outliner
    if _draw_handler_outliner is None:
        try:
            _draw_handler_outliner = bpy.types.SpaceOutliner.draw_handler_add(
                draw_outliner_axis_indicators, (), 'WINDOW', 'POST_PIXEL')
        except Exception:
            pass


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.VIEW3D_MT_object.append(menu_func)
    bpy.types.VIEW3D_MT_object_context_menu.append(context_menu_func)

    # Outliner axis indicator: register immediately so it works without ever
    # enabling preview (the 3D GPU handlers are lazy — only activated by preview).
    global _draw_handler_outliner
    if _draw_handler_outliner is None:
        try:
            _draw_handler_outliner = bpy.types.SpaceOutliner.draw_handler_add(
                draw_outliner_axis_indicators, (), 'WINDOW', 'POST_PIXEL')
        except Exception:
            pass

    # Add warning dismissal property
    bpy.types.Scene.origin_dismissed_bbox_warning = bpy.props.BoolProperty(
        name="Dismissed BBox Warning",
        default=False,
        description="User has seen the default cube bbox warning"
    )
    # UI collapse states
    bpy.types.Scene.origin_show_preview_controls = bpy.props.BoolProperty(
        name="Show Preview Controls",
        default=False,
        description="Show/hide preview customization options",
        update = origin_show_preview_controls_update
    )
    bpy.types.Scene.origin_show_general_settings = bpy.props.BoolProperty(
        name="Show General Settings",
        default=False,
        description="Show/hide general addon settings"
    )
    
    bpy.types.Scene.origin_show_faces = bpy.props.BoolProperty(
        name="Show Faces",
        default=True,
        description="Show/hide face snap buttons"
    )
    
    bpy.types.Scene.origin_show_vertices = bpy.props.BoolProperty(
        name="Show Vertices",
        default=False,
        description="Show/hide vertex snap buttons"
    )
    
    bpy.types.Scene.origin_show_edges = bpy.props.BoolProperty(
        name="Show Edges",
        default=False,
        description="Show/hide edge snap buttons"
    )
    
    bpy.types.Scene.origin_show_centers = bpy.props.BoolProperty(
        name="Show Centers",
        default=False,
        description="Show/hide center snap buttons"
    )
    
    bpy.types.Scene.origin_show_global_centers = bpy.props.BoolProperty(
        name="Show Global Centers",
        default=False,
        description="Show/hide global center snap buttons (Geometry, BBox, Mass)"
    )
    # UI Collapse ends here

    bpy.types.Scene.show_origin_preview = bpy.props.BoolProperty(
        name="Show Origin Preview",
        default=False
    )
    bpy.types.Scene.origin_preview_color = bpy.props.FloatVectorProperty(
        name="Highlight Color",
        subtype='COLOR',
        default=(1.0, 0.0, 0.0),
        min=0.0,
        max=1.0
    )
    bpy.types.Scene.origin_preview_alpha = bpy.props.FloatProperty(
        name="Highlight Alpha",
        default=0.35,
        min=0.0,
        max=1.0
    )
    bpy.types.Scene.origin_snap_source = bpy.props.EnumProperty(
        name="Snap Source",
        items=[
            ('MESH',   "Mesh (Geometry)", "Snap to evaluated mesh geometry extremes"),
            ('BBOX',   "Bounding Box",    "Snap to object bounding box extremes"),
            ('CURSOR', "3D Cursor",       "Move the origin to the 3D Cursor position (ignores geometry)"),
        ],
        default='MESH',
        update=origin_snap_source_update
    )

    # ── Axis Lock ──────────────────────────────────────────────────────────────
    bpy.types.Scene.origin_lock_x = bpy.props.BoolProperty(
        name="Lock X", default=False,
        description="Freeze the X axis — snapping will not move the origin along X"
    )
    bpy.types.Scene.origin_lock_y = bpy.props.BoolProperty(
        name="Lock Y", default=False,
        description="Freeze the Y axis — snapping will not move the origin along Y"
    )
    bpy.types.Scene.origin_lock_z = bpy.props.BoolProperty(
        name="Lock Z", default=False,
        description="Freeze the Z axis — snapping will not move the origin along Z"
    )

    # ── Numeric Offset ─────────────────────────────────────────────────────────
    bpy.types.Scene.origin_offset_x = bpy.props.FloatProperty(
        name="X Offset", default=0.0, unit='LENGTH', step=1, precision=4,
        description="World-space X offset — drag to move origin live when Live Offset is enabled",
        update=_live_offset_update_x
    )
    bpy.types.Scene.origin_offset_y = bpy.props.FloatProperty(
        name="Y Offset", default=0.0, unit='LENGTH', step=1, precision=4,
        description="World-space Y offset — drag to move origin live when Live Offset is enabled",
        update=_live_offset_update_y
    )
    bpy.types.Scene.origin_offset_z = bpy.props.FloatProperty(
        name="Z Offset", default=0.0, unit='LENGTH', step=1, precision=4,
        description="World-space Z offset — drag to move origin live when Live Offset is enabled",
        update=_live_offset_update_z
    )
    bpy.types.Scene.origin_live_offset = bpy.props.BoolProperty(
        name="Live Offset",
        default=True,
        description="When enabled, dragging the offset fields moves the origin in real time"
    )
    # Feature 3: object-space offset
    bpy.types.Scene.origin_offset_space = bpy.props.EnumProperty(
        name="Offset Space",
        items=[
            ('WORLD', "World", "Offset X/Y/Z are in world space"),
            ('LOCAL', "Local", "Offset X/Y/Z are along the object's own local axes"),
        ],
        default='WORLD',
        description="Coordinate space for the numeric offset fields"
    )
    bpy.types.Scene.origin_show_placement_tools = bpy.props.BoolProperty(
        name="Show Placement Tools",
        default=True,
        description="Show/hide the Placement Tools section (Offset, Direct, Freeze Axis, Copy/Paste)"
    )
    bpy.types.Scene.origin_copy_mode = bpy.props.EnumProperty(
        name="Copy Mode",
        items=[
            ('WORLD', "World",
             "Copy the absolute world position of the origin. "
             "Pasting places every target object\'s origin at that exact world coordinate"),
            ('DELTA', "Delta",
             "Copy the offset between this object\'s bounding box center and its origin. "
             "Pasting applies the same relative offset to each target object\'s own bbox center"),
        ],
        default='WORLD',
        description="Controls what Copy Origin stores and how Paste Origin applies it"
    )
    bpy.types.Scene.origin_offset_mode = bpy.props.EnumProperty(
        name="Offset Mode",
        items=[
            ('OFFSET', "Offset",
             "Fields are a delta added on top of the last snap point. "
             "Snap a face, then nudge the origin away from it"),
            ('DIRECT', "Direct",
             "Fields ARE the world position. Drag to place the origin at "
             "exact world coordinates — no geometry snapping required"),
        ],
        default='OFFSET',
        update=_offset_mode_update,
        description="Controls what the X/Y/Z fields represent"
    )

    # ── Edit Mode + History section visibility ─────────────────────────────────
    bpy.types.Scene.origin_show_editmode = bpy.props.BoolProperty(
        name="Show Edit Mode Snap", default=True,
        description="Show/hide the Edit Mode origin snap section"
    )
    bpy.types.Scene.origin_show_group_tools = bpy.props.BoolProperty(
        name="Show Group & Proportional Tools", default=False,
        description="Show/hide the group-center and proportional origin tools"
    )
    bpy.types.Scene.origin_show_history = bpy.props.BoolProperty(
        name="Show Origin History", default=True,
        description="Show/hide the origin history section in Preview Controls"
    )
    bpy.types.Scene.origin_show_saved_configs = bpy.props.BoolProperty(
        name="Show Saved Configurations", default=True,
        description="Show/hide the Saved Configurations section"
    )
    # Pro feature section visibility
    bpy.types.Scene.origin_show_snap_section  = bpy.props.BoolProperty(name="Show Set Origin To",      default=True)
    bpy.types.Scene.origin_show_pro_tools    = bpy.props.BoolProperty(name="Show Pro Tools",         default=True)
    bpy.types.Scene.origin_show_curve_snap   = bpy.props.BoolProperty(name="Show Curve Snap",        default=False)
    bpy.types.Scene.origin_show_obj_snap     = bpy.props.BoolProperty(name="Show Object Origin Snap", default=False)
    bpy.types.Scene.origin_show_scene_snap   = bpy.props.BoolProperty(name="Show Scene Surface Snap", default=False)
    bpy.types.Scene.origin_show_align        = bpy.props.BoolProperty(name="Show Align Origins",      default=False)
    bpy.types.Scene.origin_show_chain_paste  = bpy.props.BoolProperty(name="Show Chain Paste",        default=False)
    bpy.types.Scene.origin_show_named_groups = bpy.props.BoolProperty(name="Show Named Groups",       default=False)
    bpy.types.Scene.origin_show_csv_export   = bpy.props.BoolProperty(name="Show CSV Export",         default=False)

    # NOTE: Object Snap Tools (source/target mesh snapping, rotation alignment,
    # snap-as-group) is a Radix Pro feature — its scene properties and operators
    # are intentionally not registered here.

    # ── Tier 1/2/3 additions ──────────────────────────────────────────────────
    bpy.types.Scene.origin_normal_offset = bpy.props.FloatProperty(
        name="Normal Offset",
        default=0.0,
        step=1,
        precision=4,
        unit='LENGTH',
        description=(
            "Offset to apply along the stored snap normal. "
            "Available after any Face, Surface, or Face-Center snap"
        )
    )
    bpy.types.Scene.origin_show_viewport_handles = bpy.props.BoolProperty(
        name="Show Viewport Handles",
        default=False,
        description="Draw clickable snap-handle dots on the active object's bounding box"
    )
    bpy.types.Scene.origin_handles_snap_type = bpy.props.EnumProperty(
        name="Handle Type",
        items=[
            ('ALL',           "All",          "Show face centers, corners, and edge midpoints"),
            ('FACE_CENTERS',  "Face Centers", "6 face-center dots only"),
            ('CORNERS',       "Corners",      "8 corner dots only"),
            ('EDGE_MIDPOINTS',"Edge Mids",    "12 edge-midpoint dots only"),
            ('BBOX_CENTER',   "BBox Center",  "Single dot at bounding box center"),
        ],
        default='FACE_CENTERS',
        description="Which handle dots to show and snap to"
    )
    bpy.types.Scene.origin_snap_auto_recalc = bpy.props.BoolProperty(
        name="Auto-Update Preview",  # ← Shorter, clearer name
        default=True,
        description="Automatically refresh the face highlight and arrow when the object or scene changes. Disable for better performance with heavy meshes or large object counts"
    )
    bpy.types.Scene.origin_keep_preview_persistent = bpy.props.BoolProperty(
        name="Keep Preview Persistent",
        default=False,
        description="Keep face highlight and arrow visible for all selected objects simultaneously, even when switching selection",
        update=keep_persistent_update
    )
    bpy.types.Scene.origin_preview_orientation = bpy.props.EnumProperty(
        name="Preview Orientation",
        items=[
            ('LOCAL', "Local", "Preview aligned to object's local axes"),
            ('GLOBAL', "Global", "Preview aligned to world axes")
        ],
        default='LOCAL',
        description="Orientation mode for preview highlight and arrow",
        update=origin_preview_orientation_update
    )
    bpy.types.Scene.origin_arrow_scale = bpy.props.FloatProperty(
        name="Arrow Scale",
        default=0.4,
        min=0.01,
        max=3.0,
        step=1,
        precision=2,
        description="Arrow length as a ratio of the object's bounding box max dimension. "
                    "0.4 = 40% of the largest bbox side. Scales automatically with object size.",
        update=origin_arrow_scale_update
    )
    bpy.types.Scene.origin_arrow_color = bpy.props.FloatVectorProperty(
        name="Arrow Color",
        subtype='COLOR',
        default=(0.0, 0.0, 0.0),
        min=0.0,
        max=1.0,
        description="Global direction-arrow color used in single-object mode "
                    "(default black). In Multi-Object mode each arrow uses its "
                    "object's quad color.",
        update=origin_arrow_scale_update
    )
    bpy.types.Scene.origin_arrow_show_stroke = bpy.props.BoolProperty(
        name="Arrow Stroke",
        default=False,
        description="Draw an outline/halo behind the arrows. OFF by default — "
                    "enabling it adds one extra draw call (better performance off)",
        update=origin_arrow_scale_update
    )
    bpy.types.Scene.origin_arrow_stroke_color = bpy.props.FloatVectorProperty(
        name="Stroke Color",
        subtype='COLOR',
        default=(1.0, 1.0, 1.0),
        min=0.0, max=1.0,
        description="Color of the optional arrow stroke/halo",
        update=origin_arrow_scale_update
    )
    bpy.types.Scene.origin_arrow_stroke_width = bpy.props.FloatProperty(
        name="Stroke Width",
        default=3.0,
        min=1.0, max=12.0,
        step=10, precision=1,
        description="Thickness of the optional arrow stroke",
        update=origin_arrow_scale_update
    )
    bpy.types.Scene.origin_arrow_stroke_opacity = bpy.props.FloatProperty(
        name="Stroke Opacity",
        default=0.5,
        min=0.0, max=1.0,
        step=5, precision=2,
        description="Opacity of the optional arrow stroke",
        update=origin_arrow_scale_update
    )
    bpy.types.Scene.origin_handle_size = bpy.props.FloatProperty(
        name="Handle Size",
        default=8.5,
        min=4.0,
        max=24.0,
        step=0.5,
        precision=1,
        description="Pixel radius of the viewport bbox handle shapes (diamonds, squares, triangles). "
                    "Size is in screen pixels — consistent regardless of zoom or object distance."
    )
    bpy.types.Scene.origin_face_offset = bpy.props.FloatProperty(
        name="Face Highlight Offset",
        default=0.001,
        min=0.0,
        max=0.1,
        step=0.01,
        precision=3,
        description="Offset the face highlight quad away from the mesh surface to prevent z-fighting. Increase if you see flickering. Range: 0 – 0.1 m",
        update=origin_arrow_scale_update  # Reuse same update callback to refresh preview
    )
    bpy.types.Scene.origin_auto_show_on_hover = bpy.props.BoolProperty(
        name="Auto-Show Preview on Menu Hover",
        default=True,
        description="Automatically show preview when hovering over addon menus"
    )
    bpy.types.Scene.origin_show_bbox_wireframe = bpy.props.BoolProperty(
        name="Show BBox Wireframe",
        default=True,
        description="Show wireframe cube for bounding boxes in Multi-Object + BBox mode (both Individual and Combined)"
    )
    # Computed proxy: Multi-Object PREVIEW is derived from the live selection.
    # A get/set proxy means every existing reader gets the correct value while
    # any legacy writes are harmless no-ops — no flag can be mutated from a
    # depsgraph callback, which was the root cause of the instability.
    def _mom_get(self):
        return mom_preview_active(self)

    def _mom_set(self, value):
        # Derived — ignore writes intentionally.
        return None

    bpy.types.Scene.origin_multi_object_preview = bpy.props.BoolProperty(
        name="Multi-Object Preview",
        description="(Automatic) Active when preview is on and 2+ objects are selected",
        get=_mom_get,
        set=_mom_set,
    )
    bpy.types.Scene.origin_flip_left_right = bpy.props.BoolProperty(
        name="Flip Left/Right",
        default=False,
        description="Swap the Left and Right labels on snap buttons to match your personal perspective (object-relative vs viewport-relative)",
        update=origin_flip_left_right_update
    )
    bpy.types.Scene.origin_show_mode_indicator = bpy.props.BoolProperty(
        name="Show Mode Indicator",
        default=False,
        description="Show a small corner overlay with the current snap source and orientation"
    )
    bpy.types.Scene.origin_hide_arrow_when_muted = bpy.props.BoolProperty(
        name="Hide Arrow When Muted",
        default=True,
        description="Hide the arrow when preview is muted"
    )
    bpy.types.Scene.origin_front_axis = bpy.props.EnumProperty(
        name="Front Face Axis",
        description=(
            "Override which local axis is treated as 'front' for the highlight quad and arrow. "
            "Does not modify the mesh or rotation — only affects the preview visualization. "
            "Use this to homogenize arrow direction across objects with different orientations, "
            "or to match export conventions (e.g. UE5 uses +X as forward)"
        ),
        items=[
            ('LOCAL_Y_POS', "+Y  (Blender Default)",  "Local +Y is front — Blender's default forward axis"),
            ('LOCAL_Y_NEG', "−Y",                     "Local −Y is front"),
            ('LOCAL_X_POS', "+X  (UE5 Default)",      "Local +X is front — matches Unreal Engine's forward axis"),
            ('LOCAL_X_NEG', "−X",                     "Local −X is front"),
            ('LOCAL_Z_POS', "+Z  (Top)",               "Local +Z is front — top face"),
            ('LOCAL_Z_NEG', "−Z  (Bottom)",            "Local −Z is front — bottom face"),
        ],
        default='LOCAL_Y_POS',
        update=origin_preview_orientation_update  # reuse — triggers preview refresh
    )

    # Multi-Object Mode properties
    bpy.types.Scene.origin_mo_snap_mode = bpy.props.EnumProperty(
        name="Snap Mode",
        items=[
            ('INDIVIDUAL', "Independent Objects", "Each object snaps to its own geometry extreme independently"),
            ('COMBINED', "Combined Group", "All objects snap to the shared extreme of the entire group")
        ],
        default='INDIVIDUAL',
        description="How snapping behaves with multiple objects"
    )
    # NOTE: origin_multi_object_preview already defined at line 3344 - duplicate removed
    bpy.types.Scene.origin_mo_affect_target = bpy.props.EnumProperty(
        name="Affect Target",
        items=[
            ('ALL_SELECTED', "All Selected", "Affect all selected objects"),
            ('CUSTOM_OBJECTS', "Custom Objects", "Affect objects from custom list"),
            ('COLLECTION', "Collection", "Affect objects from selected collection")
        ],
        default='ALL_SELECTED',
        description="Which objects to affect when snapping"
    )
    # NOTE: origin_mo_custom_objects / origin_mo_target_collection are Basic/Pro
    # features (the CUSTOM_OBJECTS / COLLECTION affect-target modes). Free always
    # uses the default ALL_SELECTED mode, so those branches never execute and
    # their backing properties are intentionally not registered here.

    # NOTE: origin_mo_global_color removed — it only fed the Pro-only
    # "Apply" global-color button. Per-object colors are auto-assigned by
    # collection (get_collection_color) without any extra control needed.

    if depsgraph_update_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)

    # Register collection color change detector
    if collection_color_change_detector not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(collection_color_change_detector)

    # File-reload safety: clear stale RNA refs when a new .blend is loaded
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)

    # Register keymap for Alt+Q pie menu
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='3D View', space_type='VIEW_3D')

        # Alt + Q = Radix Pie Menu
        kmi = km.keymap_items.new(
            'wm.call_menu',
            'Q', 'PRESS',
            alt=True
        )
        kmi.properties.name = 'VIEW3D_MT_radix_pie'
        addon_keymaps.append((km, kmi))

    # NOTE: no AddonPreferences class in Radix Free, so there are no
    # preference-default timers to register here.

    # NOTE: Radix Place (radixplace_align_axis, grid step, chain step, snap
    # layers), Snap History HUD, and Viewport Mode Indicator are Basic/Pro
    # features — their scene properties and draw handlers are intentionally
    # not registered in Radix Free.


def unregister():
    global _preview_cache
    global _persistent_previews
    global _auto_show_timer
    global addon_keymaps
    global _cleanup_scheduled

    # Cancel any pending deferred-cleanup timer
    try:
        if bpy.app.timers.is_registered(_deferred_cleanup):
            bpy.app.timers.unregister(_deferred_cleanup)
    except Exception:
        pass
    _cleanup_scheduled = False

    # Remove keymaps
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    addon_keymaps.clear()

    # Cancel any pending auto-show timer
    if _auto_show_timer:
        try:
            if bpy.app.timers.is_registered(auto_show_preview_timer):
                bpy.app.timers.unregister(auto_show_preview_timer)
        except Exception:
            pass
        _auto_show_timer = None

    scene = bpy.context.scene

    _preview_cache.clear()
    _persistent_previews.clear()
    scene.show_origin_preview = False
    remove_handlers()

    # B8: Clean up Collision Preview handler
    if _collision_handler[0] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_collision_handler[0], 'WINDOW')
        except Exception:
            pass
        _collision_handler[0] = None
    _collision_active[0] = False

    # F3: Clean up Snap History HUD handler
    if _history_hud_handler[0] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_history_hud_handler[0], 'WINDOW')
        except Exception:
            pass
        _history_hud_handler[0] = None

    # F6: Clean up Mode Indicator handler
    if _mode_ind_handler[0] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_mode_ind_handler[0], 'WINDOW')
        except Exception:
            pass
        _mode_ind_handler[0] = None

    objs_to_remove = [o for o in bpy.data.objects
                      if o.name.startswith(PREVIEW_ARROW_NAME) or PREVIEW_ARROW_NAME in o.name]
    for arrow_obj in objs_to_remove:
        for col in list(arrow_obj.users_collection):
            try:
                col.objects.unlink(arrow_obj)
            except Exception:
                pass
        try:
            bpy.data.objects.remove(arrow_obj, do_unlink=True)
        except Exception:
            pass

    col = bpy.data.collections.get(PREVIEW_COLLECTION)
    if col and len(col.objects) == 0:
        try:
            bpy.context.scene.collection.children.unlink(col)
        except Exception:
            pass
        try:
            bpy.data.collections.remove(col)
        except Exception:
            pass

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    try:
        bpy.types.VIEW3D_MT_object.remove(menu_func)
    except Exception:
        pass
    try:
        bpy.types.VIEW3D_MT_object_context_menu.remove(context_menu_func)
    except Exception:
        pass

    # Remove handlers (consolidated - removed duplicate code)
    try:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update_handler)
    except Exception:
        pass

    try:
        bpy.app.handlers.depsgraph_update_post.remove(collection_color_change_detector)
    except Exception:
        pass

    try:
        bpy.app.handlers.load_post.remove(_on_load_post)
    except Exception:
        pass

    props_to_delete = [
        "show_origin_preview",
        "origin_dismissed_bbox_warning",
        "origin_show_preview_controls",
        "origin_show_general_settings",
        "origin_show_faces",
        "origin_show_vertices",
        "origin_show_edges",
        "origin_show_centers",
        "origin_show_global_centers",
        "origin_preview_color",
        "origin_preview_alpha",
        "origin_snap_source",
        "origin_keep_preview_persistent",
        "origin_arrow_scale",
        "origin_arrow_color",
        "origin_arrow_show_stroke",
        "origin_arrow_stroke_color",
        "origin_arrow_stroke_width",
        "origin_arrow_stroke_opacity",
        "origin_handle_size",
        "origin_face_offset",
        "origin_snap_auto_recalc",
        "origin_auto_show_on_hover",
        "origin_show_bbox_wireframe",
        "origin_preview_orientation",
        "origin_multi_object_preview",
        "origin_flip_left_right",
        "origin_hide_arrow_when_muted",
        "origin_mo_snap_mode",
        "origin_mo_affect_target",
        "origin_mo_custom_objects",
        "origin_mo_target_collection",
        "origin_front_axis",
        "origin_lock_x",
        "origin_lock_y",
        "origin_lock_z",
        "origin_offset_x",
        "origin_offset_y",
        "origin_offset_z",
        "origin_show_editmode",
        "origin_show_group_tools",
        "origin_show_history",
        "origin_show_snap_section",
        "origin_copy_mode",
        "origin_live_offset",
        "origin_offset_mode",
        "origin_offset_space",
        "origin_show_align",
        "origin_show_chain_paste",
        "origin_show_csv_export",
        "origin_show_curve_snap",
        "origin_show_named_groups",
        "origin_show_obj_snap",
        "origin_show_placement_tools",
        "origin_show_saved_configs",
        "origin_show_scene_snap",
        "origin_show_pro_tools",
        "origin_normal_offset",
        "origin_show_viewport_handles",
        "origin_handles_snap_type",
        # Radix Place Phase 2
        "radixplace_align_axis",
        "radixplace_snap_layers",
        # Radix Place Phase 3 + v3.1 additions
        "radixplace_grid_step",
        "radixplace_chain_step_x",
        "radixplace_chain_step_y",
        "radixplace_chain_step_z",
        "origin_show_mode_indicator",
    ]

    for prop in props_to_delete:
        try:
            delattr(bpy.types.Scene, prop)
        except Exception:
            pass


if __name__ == "__main__":
    register()