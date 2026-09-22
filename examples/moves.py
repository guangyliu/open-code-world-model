"""Unit clips: one control at a time, rendered as proxy conditions with the control overlaid.

Each clip isolates a single thing the proxy can be asked to do -- the subject walking in one
direction, the camera translating along one axis, the camera rotating about one axis, a
jump, boarding and driving a car -- so that what the conditions show can be checked against
what was commanded without anything else moving. The overlay is derived from the rendered
camera pose and subject position exactly as `mkcompare.py` does it for the released
examples, not from the command, so a clip whose overlay disagrees with its title is a bug in
the rig, not in the overlay.

    python examples/moves.py            # all clips into out/moves
    python examples/moves.py jump drive # just these
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av                                            # noqa: E402
from PIL import Image, ImageDraw, ImageFont          # noqa: E402

from compiler.render import Camera, render_frame     # noqa: E402
from world import actions as A                       # noqa: E402
from world import compile as wc                      # noqa: E402
from world import rules                              # noqa: E402
from world.scenes import StreetSpec, build, camera   # noqa: E402
from world.state import Entity                       # noqa: E402

FPS = 24
OUT = Path(os.environ.get("CWM_MOVES_OUT", "out/moves"))
S = 2
W, H, HUD = 336 * S, 192 * S, 150

COL = {0: (92, 92, 110), 1: (150, 200, 245), 2: (60, 130, 220), 4: (120, 110, 100),
       5: (55, 145, 70), 8: (255, 70, 70), 9: (255, 170, 40), 10: (250, 225, 60)}
LUT = np.zeros((256, 3), np.uint8); LUT[:] = (255, 0, 255)
for k, v in COL.items():
    LUT[k] = v


def _font(bold=False, mono=False, size=14):
    cands = {
        (False, False): ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"],
        (True, False): ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"],
        (False, True): ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"],
    }[(bold, mono)]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


f12, f13m, f14, f16b = _font(size=12), _font(mono=True, size=11), _font(size=14), _font(bold=True, size=16)
V_ON, PAN_ON, TURN_ON = 0.15, 2.0, 5.0


# ----------------------------------------------------------------------------- scene ----
def scene(seed=7):
    """The default street, rules restricted to locomotion so nothing steers the subject."""
    spec = StreetSpec(seed=seed)
    s = build(spec)
    h = s.get("courier")
    h.attrs.pop("waypoints", None)
    h.speed = 0.0            # build() starts him walking; each clip decides that itself
    return s, spec


def lens_axes(cam, subject=None):
    """The frame every key is read in: the lens's optical axis projected level (forward),
    its level perpendicular (right), and world up.

    Not the lens's own tilted basis: a −8° pitch put 14% of forward travel on the down axis
    and lit 'up' on every frame of a level walk. Not the boom line (lens -> subject) either:
    with a locked lens a subject walking across the frame swings that line, so a pure
    sideways walk read as W+A. The optical axis is what the viewer sees as 'ahead'.

    What remains is the rig's shoulder offset: it yaws the axis 10.6° toward the subject,
    so a dead-straight walk carries sin(10.6°) = 0.18 of its speed across the frame. That
    is geometry, not input, and `dominant()` keeps it dark by asking a key to carry more
    than a third of the speed before it lights.
    """
    r, d, f = cam.basis()
    f = f * np.array([1.0, 0.0, 1.0]); f /= np.linalg.norm(f)
    r = r * np.array([1.0, 0.0, 1.0]); r /= np.linalg.norm(r)
    return r, np.array([0.0, -1.0, 0.0]), f          # right, up, forward


SHARE = 0.35


def dominant(v):
    """Which of W/A/S/D a level velocity presses: a component must exceed V_ON and carry
    more than SHARE of the speed. The 10.6° offset yaw puts 0.18 across; a 45° diagonal
    puts 0.71; a sideways walk 1.0."""
    vr, vf = float(v[0]), float(v[2])
    sp = float(np.hypot(vr, vf))
    if sp < V_ON:
        return dict(W=False, A=False, S=False, D=False)
    return dict(W=vf > SHARE * sp, S=vf < -SHARE * sp, A=vr < -SHARE * sp, D=vr > SHARE * sp)


def locked_camera(spec, s, distance):
    """A lens fixed behind the subject's start, `distance` metres back."""
    ctl = camera(spec)
    ctl.distance = distance
    ctl.mode = "lock"
    h = s.get("courier")
    cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
    return ctl, cam


def cam_copy(c, **kw):
    d = dict(focal_px=c.focal_px, position=c.position.copy(), pitch_deg=c.pitch_deg,
             yaw_deg=c.yaw_deg, roll_deg=c.roll_deg)
    d.update(kw)
    return Camera(**d)


def step(s, dt=1.0 / FPS):
    rules.step(s, ["locomotion"])
    wc.advance_gait(s)


# ---------------------------------------------------------------------------- clips ----
# Each returns a list of (Camera, state_json) per frame plus a one-line command string.

def subject_walk(direction, frames=72, speed=1.5):
    s, spec = scene()
    h = s.get("courier")
    heading = {"fwd": 0.0, "back": 180.0, "left": -90.0, "right": 90.0}[direction]
    ctl, cam = locked_camera(spec, s, 4.5)
    if direction == "back":
        # Walking toward a lens 4.5 m away at 1.5 m/s reaches it in 3 s; start further out.
        h.pos = h.pos + np.array([0.0, 0.0, 3.0])
    out = []
    for k in range(frames):
        h.heading_deg, h.speed = heading, (speed if k >= 6 else 0.0)
        step(s)
        out.append((cam_copy(cam), s.to_json()))
    return out, f"subject walks {direction} at {speed} m/s (heading {heading:+.0f}°), camera LOCKED"


def subject_walk_follow(frames=96, speed=2.0):
    """Forward, a 90° right turn over one second, forward again -- the follow rig responds."""
    s, spec = scene()
    h = s.get("courier")
    ctl = camera(spec)
    out = []
    for k in range(frames):
        h.speed = speed
        if 36 <= k < 60:
            h.heading_deg += 90.0 / 24
        step(s)
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        out.append((cam_copy(cam), s.to_json()))
    return out, f"subject walks {speed} m/s, turns right 90° over 1 s at t=1.5 s, camera FOLLOW"


def camera_move(axis, frames=72, v=1.2):
    """Subject idles; the lens translates along one of its own axes."""
    s, spec = scene()
    ctl, cam0 = locked_camera(spec, s, 3.5)
    r, u, f = lens_axes(cam0)
    vec = {"fwd": f, "back": -f, "left": -r, "right": r, "up": u, "down": -u}[axis]
    out = []
    for k in range(frames):
        step(s)
        t = max(0.0, (k - 6) / FPS)
        out.append((cam_copy(cam0, position=cam0.position + vec * v * t), s.to_json()))
    return out, f"camera translates {axis} at {v} m/s (level axes of the lens), subject idle"


def camera_rotate(axis, frames=72, amp=30.0):
    """Subject idles; the lens sweeps one angle: 0 -> +amp -> -amp -> 0 over the clip."""
    s, spec = scene()
    ctl, cam0 = locked_camera(spec, s, 3.5)
    out = []
    for k in range(frames):
        step(s)
        a = amp * np.sin(2 * np.pi * k / frames)
        if axis == "pan":
            c = cam_copy(cam0, yaw_deg=cam0.yaw_deg + a)
        elif axis == "tilt":
            c = cam_copy(cam0, pitch_deg=cam0.pitch_deg + a * 0.6)
        else:
            c = cam_copy(cam0, roll_deg=cam0.roll_deg + a * 0.5)
        out.append((c, s.to_json()))
    return out, f"camera {axis} sweep ±{amp if axis == 'pan' else amp * (0.6 if axis == 'tilt' else 0.5):.0f}°, subject idle"


def jump(frames=72, v0=3.2):
    """Two standing jumps; ballistic, +Y is down so altitude is negative y."""
    s, spec = scene()
    h = s.get("courier")
    ctl, cam = locked_camera(spec, s, 4.0)
    starts = (12, 44)
    out = []
    for k in range(frames):
        h.speed = 0.0
        step(s)
        p = h.pos
        alt = 0.0
        for k0 in starts:
            t = (k - k0) / FPS
            if t >= 0:
                alt = max(alt, v0 * t - 0.5 * 9.81 * t * t)
        p[1] = -alt
        h.pos = p
        out.append((cam_copy(cam), s.to_json()))
    return out, f"two standing jumps, take-off {v0} m/s (apex {v0**2/2/9.81:.2f} m), camera LOCKED"


def drive(frames=120):
    """Walk to a car 2.5 m ahead, board it over 1 s, drive off; the follow rig rides along."""
    s, spec = scene()
    h = s.get("courier")
    car = Entity("car", "vehicle", [0.0, 0.0, 3.0], heading_deg=0.0, size=[1.9, 1.5, 4.4])
    car.attrs["arrived"] = 1.0
    s.add(car)
    rs = rules.RuleSet()
    for name, src in (("board", A.build(A.BOARD_VEHICLE, "board", vehicle="car", rider="courier",
                                         board_s=1.0, seat_frac=0.55)),
                      ("drive", A.build(A.DRIVE_OFF, "drive", vehicle="car", rider="courier",
                                         v_max=8.0, accel=3.0, seat_frac=0.55))):
        ns = {"np": np}
        exec(src, ns)
        rs.add(name, ns[name])
    ctl = camera(spec)
    out = []
    for k in range(frames):
        if k < 24:
            h.speed = 1.5
            rs.step(s, ["locomotion"])
        elif k < 48:
            h.speed = 0.0
            rs.step(s, ["locomotion", "board"])
        else:
            rs.step(s, ["locomotion", "board", "drive"])
        wc.advance_gait(s)
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        out.append((cam_copy(cam), s.to_json()))
    return out, "walk 1 s to a car, board over 1 s, drive off at 3 m/s² to 8 m/s, camera FOLLOW"


def wall_walk(frames=96, speed=1.5, solid=True):
    """A wall across the path 5 m ahead; the walk command never stops. With `blocking`
    active he stops at the face; without it (the world before today) he walks through."""
    s, spec = scene()
    h = s.get("courier")
    s.add(Entity("wall", "scenery", [0.0, 0.0, 5.0], heading_deg=0.0, size=[14.0, 4.0, 1.0]))
    ctl, cam = locked_camera(spec, s, 4.5)
    active = ["blocking", "locomotion"] if solid else ["locomotion"]
    out = []
    for k in range(frames):
        h.heading_deg, h.speed = 0.0, (speed if k >= 6 else 0.0)
        rules.step(s, active)
        wc.advance_gait(s)
        out.append((cam_copy(cam), s.to_json()))
    return out, (f"walk command held at {speed} m/s into a wall 5 m ahead, "
                 + ("blocking rule ON" if solid else "blocking rule OFF (before)") + ", camera LOCKED")


def wall_drive(frames=120, solid=True):
    """Already in the car; drive at 3 m/s² toward a wall 25 m ahead, follow rig riding."""
    s, spec = scene()
    h = s.get("courier")
    car = Entity("car", "vehicle", [0.0, 0.0, 0.0], heading_deg=0.0, size=[1.9, 1.5, 4.4])
    car.attrs["arrived"] = 1.0
    car.attrs["boarded"] = 1.0
    s.add(car)
    s.add(Entity("wall", "scenery", [0.0, 0.0, 25.0], heading_deg=0.0, size=[14.0, 4.0, 1.0]))
    h.pos = [0.0, -car.size[1] * 0.55, 0.0]
    rs = rules.RuleSet()
    ns = {"np": np}
    exec(A.build(A.DRIVE_OFF, "drive", vehicle="car", rider="courier", v_max=8.0, accel=3.0,
                 seat_frac=0.55), ns)
    rs.add("drive", ns["drive"])
    active = (["blocking"] if solid else []) + ["locomotion", "drive"]
    ctl = camera(spec)
    out = []
    for k in range(frames):
        rs.step(s, active)
        wc.advance_gait(s)
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        out.append((cam_copy(cam), s.to_json()))
    return out, ("drive at 3 m/s² into a wall 25 m ahead, "
                 + ("blocking rule ON" if solid else "blocking rule OFF (before)") + ", camera FOLLOW")


def _ramp_heading(h, target, frames_total):
    """Turn `h` toward `target` at a constant rate spread over `frames_total` frames."""
    d = ((target - h.heading_deg + 180.0) % 360.0) - 180.0
    h.heading_deg += d / frames_total


def walk_dir(direction, frames=96, speed=1.5):
    """Diagonal walks with the follow rig: heading swings ±45° over half a second, then
    holds. The boom re-centres behind him, so the background rotates while he stays put."""
    s, spec = scene()
    h = s.get("courier")
    target = {"front_left": -45.0, "front_right": 45.0}[direction]
    ctl = camera(spec)
    out = []
    for k in range(frames):
        h.speed = speed
        if 6 <= k < 18:
            _ramp_heading(h, target, 18 - k)
        step(s)
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        out.append((cam_copy(cam), s.to_json()))
    return out, f"subject walks {direction.replace('_', '-')} ({target:+.0f}°) at {speed} m/s, camera FOLLOW"


def drive_dir(direction, frames=96):
    """In the car from frame 0, follow rig on the rider. A car cannot strafe, so left/right
    are 90° turns and the diagonals are 45° turns, all at a constant road speed; back is
    reverse gear. The turn radius is v / rate: 5 m/s at 60°/s is 4.8 m."""
    s, spec = scene()
    h = s.get("courier")
    car = Entity("car", "vehicle", [0.0, 0.0, 0.0], heading_deg=0.0, size=[1.9, 1.5, 4.4])
    car.attrs["arrived"] = car.attrs["boarded"] = 1.0
    s.add(car)
    seat = car.size[1] * 0.55
    v = {"fwd": 5.0, "back": -3.0, "left": 5.0, "right": 5.0, "front_left": 5.0, "front_right": 5.0}[direction]
    target = {"fwd": 0.0, "back": 0.0, "left": -90.0, "right": 90.0, "front_left": -45.0, "front_right": 45.0}[direction]
    turn_frames = {0.0: 0, 45.0: 18, 90.0: 36}[abs(target)]
    ctl = camera(spec)
    out = []
    for k in range(frames):
        car.speed = v if k >= 6 else 0.0
        if 12 <= k < 12 + turn_frames:
            _ramp_heading(car, target, 12 + turn_frames - k)
        rules.step(s, ["blocking", "locomotion"])
        # Rider rides: same as DRIVE_OFF's seat placement, without its throttle.
        cp = car.pos
        h.pos = [cp[0], -seat, cp[2]]
        h.heading_deg, h.speed = car.heading_deg, 0.0
        wc.advance_gait(s)
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        out.append((cam_copy(cam), s.to_json()))
    what = {"fwd": "drives forward", "back": "reverses", "left": "turns left 90°", "right": "turns right 90°",
            "front_left": "veers front-left 45°", "front_right": "veers front-right 45°"}[direction]
    return out, f"car {what} at {abs(v)} m/s, rider aboard, camera FOLLOW"


def knockdown(frames=120, speed=1.5):
    """Walk up to a pedestrian standing in the way; one hit within 1 m and he goes down in
    0.5 s and stays down; the walker pauses over him, then carries on past."""
    s, spec = scene()
    h = s.get("courier")
    # Half a stride to the right of the walker's line, so the fall is not hidden behind him
    # from a lens that sits on his shoulder; he then walks on past the body.
    victim = Entity("victim", "human", [0.8, 0.0, 4.5], heading_deg=180.0, size=[0.55, 1.75, 0.40])
    s.add(victim)
    rs = rules.RuleSet()
    ns = {"np": np}
    exec(A.build(A.KNOCKDOWN, "hit", attacker="courier", target="victim", reach=1.2, pause_s=0.8), ns)
    rs.add("hit", ns["hit"])
    ctl = camera(spec)
    ctl.distance = 4.0
    out = []
    for k in range(frames):
        if not h.attrs.get("hit_pause") or h.attrs["hit_pause"] <= 0.0:
            h.heading_deg, h.speed = 0.0, (speed if k >= 6 else 0.0)
        rs.step(s, ["hit", "falling", "blocking", "locomotion"])
        wc.advance_gait(s)
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        out.append((cam_copy(cam), s.to_json()))
    return out, f"walk at {speed} m/s up to a pedestrian 4.5 m ahead; one hit within 1.2 m, he falls in 0.5 s and stays down; walk on past, camera FOLLOW"


def runover(frames=120, speed=8.0):
    """In the car at 8 m/s; a pedestrian stands on the line 20 m ahead, slightly right of
    centre. The `impact` rule launches him; the car brakes at 6 m/s² after the hit."""
    s, spec = scene()
    h = s.get("courier")
    car = Entity("car", "vehicle", [0.0, 0.0, 0.0], heading_deg=0.0, size=[1.9, 1.5, 4.4])
    car.attrs["arrived"] = car.attrs["boarded"] = 1.0
    s.add(car)
    seat = car.size[1] * 0.55
    h.attrs["riding"] = "car"
    s.add(Entity("victim", "human", [0.4, 0.0, 20.0], heading_deg=180.0, size=[0.55, 1.75, 0.40]))
    ctl = camera(spec)
    ctl.distance = 4.5
    out, hit_at = [], None
    for k in range(frames):
        if hit_at is None:
            car.speed = speed if k >= 4 else 0.0
        else:
            car.speed = max(0.0, car.speed - 6.0 / FPS)
        rules.step(s, ["impact", "falling", "blocking", "locomotion"])
        if hit_at is None and any(e.kind == "run_over" for e in s.events):
            hit_at = k
        cp = car.pos
        h.pos = [cp[0], -seat, cp[2]]
        h.heading_deg, h.speed = car.heading_deg, 0.0
        wc.advance_gait(s)
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        out.append((cam_copy(cam), s.to_json()))
    return out, f"car at {speed} m/s into a pedestrian 20 m ahead; he is launched, tumbles and lands; car brakes at 6 m/s², camera FOLLOW"


CLIPS = {
    "knockdown": knockdown,
    "runover": runover,
    "drive_fwd": lambda: drive_dir("fwd"),
    "drive_back": lambda: drive_dir("back"),
    "drive_left": lambda: drive_dir("left"),
    "drive_right": lambda: drive_dir("right"),
    "drive_front_left": lambda: drive_dir("front_left"),
    "drive_front_right": lambda: drive_dir("front_right"),
    "walk_front_left": lambda: walk_dir("front_left"),
    "walk_front_right": lambda: walk_dir("front_right"),
    "wall_walk_before": lambda: wall_walk(solid=False),
    "wall_walk": lambda: wall_walk(solid=True),
    "wall_drive_before": lambda: wall_drive(solid=False),
    "wall_drive": lambda: wall_drive(solid=True),
    "walk_fwd": lambda: subject_walk("fwd"),
    "walk_back": lambda: subject_walk("back"),
    # 1.0 m/s: at 1.5 the subject crossed the 3.7 m half-width of a 4.5 m stand-off and
    # left the frame on the near side before the clip ended.
    "walk_left": lambda: subject_walk("left", speed=1.0),
    "walk_right": lambda: subject_walk("right", speed=1.0),
    "walk_turn_follow": subject_walk_follow,
    "cam_fwd": lambda: camera_move("fwd"),
    "cam_back": lambda: camera_move("back"),
    "cam_left": lambda: camera_move("left"),
    "cam_right": lambda: camera_move("right"),
    "cam_up": lambda: camera_move("up", v=0.6),
    "cam_pan": lambda: camera_rotate("pan"),
    "cam_tilt": lambda: camera_rotate("tilt"),
    "cam_roll": lambda: camera_rotate("roll"),
    "jump": jump,
    "drive": drive,
}


# -------------------------------------------------------------------------- overlay ----
def controls(plan):
    b0, b1 = Camera(yaw_deg=0).basis(), Camera(yaw_deg=1).basis()
    yaw_pos_is_right = float(b0[0] @ b1[2]) > 0
    sgn = 1 if yaw_pos_is_right else -1
    out, prev = [], None
    for k, (cam, sj) in enumerate(plan):
        st = json.loads(sj)
        h = st["entities"]["courier"]
        rec = dict(k=k, pos=cam.position.copy(), pitch=cam.pitch_deg, yaw=cam.yaw_deg,
                   roll=cam.roll_deg, hpos=np.array(h["position"], float),
                   hhead=float(h["heading_deg"]),
                   dist=float(np.linalg.norm(cam.position - np.array(h["position"], float))))
        if prev is None:
            rec.update(v=np.zeros(3), yaw_rate=0.0, pitch_rate=0.0, roll_rate=0.0, turn=0.0,
                       climb=0.0, hspeed=0.0)
        else:
            r_, u_, f_ = lens_axes(cam)
            vw = (rec["pos"] - prev["pos"]) * FPS
            rec["v"] = np.array([vw @ r_, -(vw @ u_), vw @ f_])   # right, down, forward
            dy = ((cam.yaw_deg - prev["yaw"] + 180) % 360) - 180
            rec["yaw_rate"] = dy * FPS * sgn
            rec["pitch_rate"] = (cam.pitch_deg - prev["pitch"]) * FPS
            rec["roll_rate"] = (cam.roll_deg - prev["roll"]) * FPS
            dh = ((rec["hhead"] - prev["hhead"] + 180) % 360) - 180
            rec["turn"] = dh * FPS * sgn
            d = rec["hpos"] - prev["hpos"]
            rec["climb"] = -d[1] * FPS
            rec["hspeed"] = float(np.hypot(d[0], d[2])) * FPS
            # Which way the subject walks RELATIVE TO THE LENS, for the title check.
            rec["hv_cam"] = np.array([d @ r_, -(d @ u_), d @ f_]) * FPS
        out.append(rec)
        prev = rec
    return out


def key(d, xy, label, on, size=30):
    x, y = xy
    fill = (255, 214, 64) if on else (58, 60, 74)
    fg = (20, 20, 24) if on else (150, 152, 166)
    d.rounded_rectangle([x, y, x + size, y + size], 6, fill=fill,
                        outline=(255, 230, 140) if on else (80, 82, 98), width=2)
    fnt = f16b if len(label) == 1 else _font(bold=True, size=12)
    tw = d.textlength(label, font=fnt)
    d.text((x + size / 2 - tw / 2, y + (5 if len(label) == 1 else 8)), label, font=fnt, fill=fg)


def hud(r, name, cmd, zsub):
    im = Image.new("RGB", (W, HUD), (24, 25, 32))
    d = ImageDraw.Draw(im)
    vr, vd, vf = (float(x) for x in r["v"])
    d.text((10, 6), f"{name}   frame {r['k']:3d}  t={r['k']/FPS:4.2f}s", font=f16b, fill=(230, 232, 244))
    d.text((10, 26), cmd, font=f12, fill=(255, 214, 64))
    y0 = 46
    hr, hd, hf = (float(x) for x in r.get("hv_cam", np.zeros(3)))
    # Subject: WASD is HIS movement as seen from the lens (W = away from the lens along
    # the boom line, A/D = across it, S = toward it), like a game's player input.
    # Plain letters throughout: the Mac fonts at hand have no ⇧/↶/⤒ glyphs and drew tofu.
    d.text((10, y0), "subject  W away · S toward · A/D across", font=f12, fill=(170, 172, 186))
    kk = dominant(r.get("hv_cam", np.zeros(3)))
    key(d, (42, y0 + 16), "W", kk["W"]); key(d, (10, y0 + 50), "A", kk["A"])
    key(d, (42, y0 + 50), "S", kk["S"]); key(d, (74, y0 + 50), "D", kk["D"])
    key(d, (120, y0 + 16), "TL", r["turn"] < -TURN_ON); key(d, (152, y0 + 16), "TR", r["turn"] > TURN_ON)
    key(d, (120, y0 + 50), "UP" if r["climb"] >= 0 else "DN", abs(r["climb"]) > V_ON)
    d.text((10, y0 + 86), f"spd {r['hspeed']:4.2f} trn {r['turn']:+5.1f} clb {r['climb']:+4.2f}", font=f13m, fill=(200, 202, 214))
    # Camera body: its own translation in the same frame, letters F/B/L/R so it cannot be
    # read as the subject's keys.
    d.text((235, y0), "camera moves", font=f12, fill=(170, 172, 186))
    kc = dominant(r["v"])
    key(d, (267, y0 + 16), "F", kc["W"]); key(d, (235, y0 + 50), "L", kc["A"])
    key(d, (267, y0 + 50), "B", kc["S"]); key(d, (299, y0 + 50), "R", kc["D"])
    key(d, (335, y0 + 16), "UP", vd < -V_ON); key(d, (335, y0 + 50), "DN", vd > V_ON)
    d.text((235, y0 + 86), f"fwd {vf:+5.2f} lat {vr:+5.2f} up {-vd:+5.2f}", font=f13m, fill=(200, 202, 214))
    d.text((455, y0), "camera turns", font=f12, fill=(170, 172, 186))
    key(d, (487, y0 + 16), "↑", r["pitch_rate"] > PAN_ON); key(d, (455, y0 + 50), "←", r["yaw_rate"] < -PAN_ON)
    key(d, (487, y0 + 50), "↓", r["pitch_rate"] < -PAN_ON); key(d, (519, y0 + 50), "→", r["yaw_rate"] > PAN_ON)
    key(d, (555, y0 + 16), "RL", abs(r["roll_rate"]) > PAN_ON)
    d.text((455, y0 + 86), f"yaw {r['yaw_rate']:+5.1f} pit {r['pitch_rate']:+5.1f} rol {r['roll_rate']:+4.1f}", font=f13m, fill=(200, 202, 214))
    d.text((595, y0 + 50), f"dist {r['dist']:4.2f}", font=f13m, fill=(220, 222, 234))
    d.text((595, y0 + 66), f"meas {zsub:4.2f}", font=f13m, fill=(255, 120, 120))
    return np.array(im)


def depth_img(z):
    g = np.zeros(z.shape + (3,), np.uint8)
    hit = z > 0
    t = np.clip(np.log(np.maximum(z, 0.3) / 0.3) / np.log(256 / 0.3), 0, 1)
    v = (235 - 215 * t).astype(np.uint8)
    g[hit] = np.stack([v, v, v], -1)[hit]
    g[~hit] = (34, 38, 52)
    return g


def up(a):
    return np.repeat(np.repeat(a, S, 0), S, 1)


def write(frames, path):
    h, w = frames[0].shape[:2]
    c = av.open(str(path), mode="w", options={"movflags": "+faststart"})
    st = c.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": "23", "preset": "medium"}
    for f in frames:
        for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()


def run(name):
    from world.state import WorldState
    plan, cmd = CLIPS[name]()
    ctl = controls(plan)
    frames, sem_all, dep_all = [], [], []
    cache = wc.PrimitiveCache()
    for (cam, sj), r in zip(plan, ctl):
        st = WorldState.from_json(sj)
        z, ids = render_frame(cam, cache.build(st))
        m = (ids == 8) & (z > 0)
        zsub = float(np.median(z[m])) if m.sum() > 20 else float("nan")
        frames.append(np.concatenate([up(LUT[ids]), up(depth_img(z)), hud(r, name, cmd, zsub)], 0))
        sem_all.append(ids); dep_all.append(z)
    OUT.mkdir(parents=True, exist_ok=True)
    write(frames, OUT / f"{name}.mp4")
    n = len(frames)
    sheet = np.concatenate([np.concatenate([up(LUT[sem_all[i]]), up(depth_img(dep_all[i]))], 0)
                            for i in (0, n // 3, 2 * n // 3, n - 1)], 1)
    Image.fromarray(sheet).save(OUT / f"{name}_sheet.png")
    # The numbers the title claims, measured from the render.
    on = lambda f: int(sum(1 for r in ctl if f(r)))
    hv = np.array([r.get("hv_cam", np.zeros(3)) for r in ctl])
    summ = dict(
        frames=n, cmd=cmd,
        cam_W=on(lambda r: dominant(r["v"])["W"]), cam_S=on(lambda r: dominant(r["v"])["S"]),
        cam_A=on(lambda r: dominant(r["v"])["A"]), cam_D=on(lambda r: dominant(r["v"])["D"]),
        cam_up=on(lambda r: r["v"][1] < -V_ON), cam_down=on(lambda r: r["v"][1] > V_ON),
        pan_l=on(lambda r: r["yaw_rate"] < -PAN_ON), pan_r=on(lambda r: r["yaw_rate"] > PAN_ON),
        tilt_u=on(lambda r: r["pitch_rate"] > PAN_ON), tilt_d=on(lambda r: r["pitch_rate"] < -PAN_ON),
        roll=on(lambda r: abs(r["roll_rate"]) > PAN_ON),
        W=on(lambda r: dominant(r.get("hv_cam", np.zeros(3)))["W"]), S=on(lambda r: dominant(r.get("hv_cam", np.zeros(3)))["S"]),
        A=on(lambda r: dominant(r.get("hv_cam", np.zeros(3)))["A"]), D=on(lambda r: dominant(r.get("hv_cam", np.zeros(3)))["D"]),
        walk=on(lambda r: r["hspeed"] > V_ON), turn_l=on(lambda r: r["turn"] < -TURN_ON),
        turn_r=on(lambda r: r["turn"] > TURN_ON), climb=on(lambda r: r["climb"] > V_ON),
        subject_v_in_lens_frame_mean=[round(float(x), 2) for x in hv[6:].mean(0)],
        max_alt=round(float(max(-r["hpos"][1] for r in ctl)), 2),
        heading=[round(ctl[0]["hhead"], 1), round(ctl[-1]["hhead"], 1)],
        cam_yaw=[round(ctl[0]["yaw"], 1), round(ctl[-1]["yaw"], 1)],
        path_xz=[round(float(ctl[-1]["hpos"][0] - ctl[0]["hpos"][0]), 2),
                 round(float(ctl[-1]["hpos"][2] - ctl[0]["hpos"][2]), 2)],
        dist=[round(min(r["dist"] for r in ctl), 2), round(max(r["dist"] for r in ctl), 2)],
        class8_frac=[round(float((s == 8).mean()), 4) for s in (sem_all[0], sem_all[-1])],
        class8_cx=[round(float(np.nonzero(s == 8)[1].mean() / 336), 3) if (s == 8).any() else None
                   for s in (sem_all[0], sem_all[-1])],
    )
    last = json.loads(plan[-1][1])
    if "victim" in last["entities"]:
        v = last["entities"]["victim"]
        alts = [-json.loads(sj)["entities"]["victim"]["position"][1] for _, sj in plan]
        summ["victim"] = dict(down=bool(v["attrs"].get("down")), down_frac=v["attrs"].get("down_frac"),
                              max_alt=round(max(alts), 2), final_xz=[round(v["position"][0], 2), round(v["position"][2], 2)],
                              events=[(e["kind"], round(e["t"], 2), e.get("detail")) for e in last["events"] if e["kind"] in ("hit", "down", "run_over", "landed")])
    if "wall" in last["entities"]:
        wz = last["entities"]["wall"]["position"][2] - last["entities"]["wall"]["size"][2] / 2
        mover = last["entities"].get("car", last["entities"]["courier"])
        summ["wall_face_z"] = round(wz, 2)
        summ["mover_final_z"] = round(mover["position"][2], 2)
        summ["gap_to_wall_m"] = round(wz - mover["position"][2] - mover["size"][2] / 2, 2)
        summ["blocked_events"] = [e for e in last["events"] if e["kind"] == "blocked"]
    return name, summ


def main():
    import multiprocessing as mp
    names = sys.argv[1:] or list(CLIPS)
    bad = [n for n in names if n not in CLIPS]
    if bad:
        print(f"unknown clip(s) {bad}; valid: {', '.join(CLIPS)}", file=sys.stderr)
        return 2
    with mp.get_context("spawn").Pool(min(8, len(names))) as pool:
        res = dict(pool.map(run, names))
    OUT.mkdir(parents=True, exist_ok=True)
    # Merge, so rendering a subset does not drop the other clips' numbers from the summary.
    prev = json.load(open(OUT / "summary.json")) if (OUT / "summary.json").exists() else {}
    prev.update(res)
    json.dump(prev, open(OUT / "summary.json", "w"), indent=1)
    for n in names:
        print(n, json.dumps(res[n]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
