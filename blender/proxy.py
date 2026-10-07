"""The simplest Blender proxy: an empty scene, a floor, random boxes, a box-and-ball
character walking forward, a camera trailing it, rendered as Workbench grey clay.

    conda activate pilot                      # bpy 5.0.1 as a Python module
    python blender/proxy.py out/blender/walk  # writes frames/f_0001.png ... and clay.mp4

Workbench needs OpenGL (EGL), so run it on a GPU node; the CPU nodes have no libEGL.
CWM_BL_FRAMES overrides the frame count for a quick test.
Output is 896x512 (480p-class, both sides divisible by 64) and 121 frames at 24 fps
(8k+1), what LTX-2.5's Layout-To-Render IC-LoRA takes.

Blender is Z-up: the character walks along +Y on the z = 0 floor.
"""
from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

# Headless GL: on these GPU nodes glvnd picks Mesa's EGL first, and Workbench then renders
# every pixel transparent black at ~20 s a frame. NVIDIA's EGL renders in milliseconds.
# Must be set before bpy is imported.
_NV_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(_NV_EGL):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NV_EGL)

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

W, H, FPS = 896, 512, 24
LANE = 7.0            # half-width of the box-free corridor, metres
LENS_MM = 28.0        # Blender's default 50 mm framed the character as a pillar
N = int(os.environ.get("CWM_BL_FRAMES", 121))


def cube(x, y, z):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
    return bpy.context.active_object


def sphere(x, y, z, r):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x, y, z), segments=32, ring_count=16)
    bpy.ops.object.shade_smooth()
    return bpy.context.active_object


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def build():
    # 1) empty scene: no default cube, camera or light
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 2) floor
    bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, 0))

    # 3) 24 boxes as "buildings", fixed seed. Boxes are pushed out of |x| < LANE: the
    # character swings to |x| = 1.6 and the camera trails 4.2 m to its left, so anything
    # nearer the axis is walked through or swallows the lens.
    random.seed(7)
    for _ in range(24):
        x, y = random.uniform(-16, 16), random.uniform(4, 34)
        sz = random.uniform(2.0, 9.0)
        sx, sy = random.uniform(1.5, 5.0), random.uniform(1.5, 5.0)
        if abs(x) < LANE + sx / 2:
            x = math.copysign(LANE + sx / 2 + abs(x) * 0.5, x if x else 1.0)
        b = cube(x, y, sz / 2)
        b.scale = (sx, sy, sz)

    # 4) character = root empty + body box + head ball, both parented to the root, so
    # moving the root moves the whole body. No limbs, nothing articulates.
    bpy.ops.object.empty_add(location=(0, 0, 0))
    char = bpy.context.active_object
    body = cube(0, 0, 0.9)
    body.scale = (0.5, 0.35, 1.4)
    body.parent = char
    head = sphere(0, 0, 1.85, 0.35)
    head.parent = char

    # 5) camera
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens = LENS_MM
    scene.camera = cam

    # 6) Workbench grey clay
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.78, 0.78, 0.80)
    sh.show_cavity = True
    sh.background_type = "VIEWPORT"
    sh.background_color = (0.05, 0.05, 0.07)
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.view_settings.view_transform = "Standard"
    return scene, char, cam


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/blender/walk").resolve()
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    scene, char, cam = build()

    # 7) per frame, positions from formulas -- no keyframes
    for f in range(1, N + 1):
        t = (f - 1) / FPS
        cx = math.sin(t * 1.1) * 1.6          # gentle side to side
        cy = t * 1.4                          # forward
        bob = 0.04 * abs(math.sin(t * 2 * math.pi * 1.8))
        char.location = (cx, cy, bob)
        cam.location = (cx - 4.2, cy - 4.4, 3.2)   # trailing, behind and to the side
        look_at(cam, (cx, cy + 1.2, 1.0))           # aim at the chest
        scene.render.filepath = str(frames / f"f_{f:04d}.png")
        bpy.ops.render.render(write_still=True)

    ff = shutil.which("ffmpeg")
    if ff:
        subprocess.run([ff, "-v", "error", "-y", "-framerate", str(FPS), "-i", str(frames / "f_%04d.png"),
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out / "clay.mp4")],
                       check=True)
    print(f"{N} frames {W}x{H} -> {out}" + ("" if ff else "  (no ffmpeg on this node: frames only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
