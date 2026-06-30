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


# ── Stubs for functions called by OBJECT_OT_set_origin_extreme_full ──────────
# These replace the full Basic/Pro implementations with minimal no-op or
# pass-through equivalents correct for the Free feature set.

def _record_history(obj):
    pass  # no per-object undo history exposed in Free UI


def push_snap_history(obj_name, world_pos):
    pass  # no Snap History in Free


def _apply_lock_and_offset(obj, target, scene):
    """No offsets or axis locks in Free — return the target position unchanged."""
    return target


def _apply_origin(obj, world_pos, context):
    """Set origin to world_pos. Simplified for Free (no offsets/history)."""
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
    finally:
        context.scene.cursor.location = saved_cursor
        deselect_all_objects(context)
        for o in saved_sel:
            if o and o.name in context.view_layer.objects:
                o.select_set(True)
        if saved_active and saved_active.name in context.view_layer.objects:
            context.view_layer.objects.active = saved_active


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
PREVIEW_ARROW_NAME = "RadixPro_Preview_Arrow"  # kept for any lingering ref in 39-pos op
PREVIEW_COLLECTION = "RadixPro_Preview_Col"

# ── Draw handler for BBox handle shapes (the only GPU callback in Free) ────────
_draw_handler_2d = None   # POST_PIXEL for shaped bbox handles

# ── Hover state: world pos of hovered handle, written by click_bbox_handle ─────
_handle_hover_world_pos: list = [None]  # None | Vector

# ── Misc constants referenced by the 39-position operator ─────────────────────
_origin_history: dict = {}       # not exposed in Free UI but operator reads it
ORIGIN_HISTORY_MAX    = 5
_origin_clipboard: dict = {}
_origin_proportion_clipboard: dict = {}
_last_snap_target     = {}
_last_snap_normals: dict = {}
_last_active_object_name = None
_cleanup_scheduled    = False
_auto_show_timer      = None
_depsgraph_updating   = False
_surface_snap_active  = False
_surface_snap_preview_point = None

# Preview cache stubs — the 39-pos operator references these internally
_preview_cache = {
    'quad_world': None, 'quad_local': None, 'obj_name': None,
    'orientation': None, 'front_axis': None, 'bbox_local': None,
}
_persistent_previews = {}



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


# ── Deferred cleanup (NEVER mutate blend data from a draw or depsgraph handler) ──

def _tag_view3d_redraw():
    """Tag all 3D viewports for redraw (robust to a missing context.screen)."""
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


# ---------------- GPU draw handler ----------------

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


# ---------------- Preview management (cache + handlers) ----------------


def ensure_handlers():
    """Register the BBox-handle-shapes draw handler. Free has no preview overlay."""
    global _draw_handler_2d
    if _draw_handler_2d is None:
        _draw_handler_2d = bpy.types.SpaceView3D.draw_handler_add(
            draw_handle_shapes_2d, (), 'WINDOW', 'POST_PIXEL')


def remove_handlers():
    """Remove the BBox-handle-shapes draw handler."""
    global _draw_handler_2d
    if _draw_handler_2d is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler_2d, 'WINDOW')
        except Exception:
            pass
        _draw_handler_2d = None

# ---------------- Depsgraph Handler ----------------

# ---------------- Callbacks for property updates ----------------


def origin_flip_left_right_update(self, context):
    """Update callback: force UI redraw when flip toggle changes."""
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
        for item in getattr(scene, "origin_mo_custom_objects", []):
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
        col = getattr(scene, "origin_mo_target_collection", None)
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


# ---------------- Operators ----------------


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
                if not getattr(scene, "origin_mo_target_collection", None):
                    self.report({'WARNING'}, "No collection selected — pick a target collection in Set Origin To panel")
                else:
                    self.report({'WARNING'}, f"Collection '{getattr(scene, "origin_mo_target_collection", None).name}' has no mesh objects")
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
            if getattr(scene, "show_origin_preview", False):
                pass  # preview not available in Radix Free

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
                getattr(context.scene, "origin_offset_x", 0.0),
                getattr(context.scene, "origin_offset_y", 0.0),
                getattr(context.scene, "origin_offset_z", 0.0),
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
                    pos.x - getattr(context.scene, "origin_offset_x", 0.0),
                    pos.y - getattr(context.scene, "origin_offset_y", 0.0),
                    pos.z - getattr(context.scene, "origin_offset_z", 0.0),
                )))
                # Direct-mode offset sync omitted in Free (no offset UI)

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

        if getattr(scene, "show_origin_preview", False):
            pass  # preview not available in Radix Free

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


# ── Collision Preview helpers ──────────────────────────────────────────────────


# ── Surface Align helper ───────────────────────────────────────────────────────


# ══ Phase 1 Operators ═════════════════════════════════════════════════════════


# ══ Phase 2 Operators ═════════════════════════════════════════════════════════


# ══ Radix Place Panel ═════════════════════════════════════════════════════════




# ══ F3: Snap History HUD draw function ════════════════════════════════════════


# ══ F6: Viewport Mode Indicator draw function ══════════════════════════════════


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
    """Reset volatile in-memory state when a new .blend file loads."""
    _handle_hover_world_pos[0] = None
    remove_handlers()


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.app.handlers.load_post.append(_on_load_post)

    # ── Scene properties needed by Free ────────────────────────────────────────
    # Snap Reference controls — source, orientation, front axis
    bpy.types.Scene.origin_snap_source = bpy.props.EnumProperty(
        name="Snap Source", default='MESH',
        items=[
            ('MESH', "Mesh",   "Snap to vertices and face centers of the mesh"),
            ('BBOX', "BBox",   "Snap to bounding box extremes"),
            ('CURSOR', "Cursor", "Snap to the 3D cursor position"),
        ],
    )
    bpy.types.Scene.origin_preview_orientation = bpy.props.EnumProperty(
        name="Orientation", default='LOCAL',
        items=[
            ('LOCAL',  "Local",  "Use the object's local axes"),
            ('GLOBAL', "Global", "Use world axes"),
        ],
    )
    bpy.types.Scene.origin_front_axis = bpy.props.EnumProperty(
        name="Front Face Axis", default='LOCAL_Y_POS',
        items=[
            ('LOCAL_Y_POS', "+Y (Forward)",  ""),
            ('LOCAL_Y_NEG', "-Y (Back)",     ""),
            ('LOCAL_X_POS', "+X (Right)",    ""),
            ('LOCAL_X_NEG', "-X (Left)",     ""),
            ('LOCAL_Z_POS', "+Z (Up)",       ""),
            ('LOCAL_Z_NEG', "-Z (Down)",     ""),
        ],
    )
    # BBox Handle mode
    bpy.types.Scene.origin_show_viewport_handles = bpy.props.BoolProperty(
        name="Show Viewport Handles", default=False)
    bpy.types.Scene.origin_handles_snap_type = bpy.props.EnumProperty(
        name="Handle Type", default='BBOX',
        items=[
            ('BBOX',   "BBox",   ""),
            ('MESH',   "Mesh",   ""),
            ('CURSOR', "Cursor", ""),
        ],
    )
    # Flip Left/Right labels
    bpy.types.Scene.origin_flip_left_right = bpy.props.BoolProperty(
        name="Flip Left/Right", default=False,
        update=origin_flip_left_right_update)
    # Multi-object affect-target (defaults to ALL_SELECTED — always used)
    bpy.types.Scene.origin_mo_affect_target = bpy.props.EnumProperty(
        name="Affect Target", default='ALL_SELECTED',
        items=[
            ('ALL_SELECTED',   "All Selected",   ""),
            ('CUSTOM_OBJECTS', "Custom Objects",  ""),
            ('COLLECTION',     "Collection",      ""),
        ],
    )
    bpy.types.Scene.origin_mo_snap_mode = bpy.props.EnumProperty(
        name="Snap Mode", default='INDIVIDUAL',
        items=[
            ('INDIVIDUAL', "Individual", ""),
            ('COMBINED',   "Combined",   ""),
        ],
    )

    # ── Register keymaps ────────────────────────────────────────────────────────
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km  = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new('wm.call_menu', 'Q', 'PRESS', alt=True)
        kmi.properties.name = 'VIEW3D_MT_radix_pie'
        addon_keymaps.append((km, kmi))

    # ── Object menu entries ─────────────────────────────────────────────────────
    bpy.types.VIEW3D_MT_object.append(menu_func)
    bpy.types.VIEW3D_MT_object_context_menu.append(context_menu_func)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    bpy.types.VIEW3D_MT_object_context_menu.remove(context_menu_func)

    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    addon_keymaps.clear()

    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)

    remove_handlers()

    props_to_delete = [
        "origin_snap_source", "origin_preview_orientation", "origin_front_axis",
        "origin_show_viewport_handles", "origin_handles_snap_type",
        "origin_flip_left_right", "origin_mo_affect_target", "origin_mo_snap_mode",
    ]
    for prop in props_to_delete:
        try:
            delattr(bpy.types.Scene, prop)
        except Exception:
            pass

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
