"""Drive the proxy live from the keyboard, GTA-style, and record the session.

    python examples/play.py                       # open a window, play, Esc to quit
    python examples/play.py --replay out/play/<stamp>/inputs.json --out out/play/<stamp>/conditions

Controls (the camera always follows; the subject moves RELATIVE TO THE VIEW):
    W A S D      walk away / left / toward / right of the lens; the follow rig translates
    Shift        run
    ← →          orbit the lens around the subject (yaw), persists until R
    ↑ ↓          tilt the lens (pitch trim)
    Space        jump
    F            hit the person in front of you (within 1.2 m); one hit and they go down
    E            get into / out of the car when within 2.5 m of it; in the car W/S are
                 throttle and brake (S reverses when stopped), A/D steer, Shift is a heavier foot.
                 A car above 2 m/s launches anyone it hits.
    R            re-centre the lens behind the subject
    Esc / Q      quit and write inputs.json

The live window renders straight at 336x192 (no 672x384 supersample + pool) because that
is ~4x faster and this is a preview. The recorded inputs replay through the same step
function at a fixed 1/24 s, so `--replay` reproduces the session exactly and writes the
published-format conditions (672x384 rendered, pooled to 336x192) that a video job eats.
Nothing about the simulation depends on wall-clock time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler import writer                                  # noqa: E402
from compiler.render import Camera, render, render_frame     # noqa: E402
from world import compile as wc                              # noqa: E402
from world import rules                                      # noqa: E402
from world.scenes import StreetSpec, build, camera           # noqa: E402
from world.state import Entity                               # noqa: E402
import moves                                                 # noqa: E402  (HUD + LUT)

FPS = 24
DT = 1.0 / FPS
WALK, RUN = 1.6, 4.0
ORBIT_RATE, TILT_RATE = 90.0, 45.0        # deg/s while the arrow is held
RECENTRE_TAU = 1.2                        # s; while moving, the view azimuth eases behind the subject
JUMP_V0, G = 3.2, 9.81
KEYS = ("W", "A", "S", "D", "SHIFT", "LEFT", "RIGHT", "UP", "DOWN", "SPACE", "R", "F", "E")
CAR_ACCEL, CAR_BRAKE, CAR_DRAG, CAR_VMAX, CAR_REV, STEER_DEG_S = 4.0, 9.0, 1.2, 14.0, -3.0, 70.0
N_PEDS = 10


class Session:
    """The whole state one frame needs, so live play and replay share one step()."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.spec = StreetSpec(seed=seed)
        self.s = build(self.spec)
        self.h = self.s.get("courier")
        self.h.attrs.pop("waypoints", None)
        self.h.speed = 0.0
        # A car to take, parked just ahead-right, and a street with people on it. The
        # pedestrians walk fixed beats between two ends of the block under the built-in
        # rules; they are launched by a fast car and knocked down by F like anyone else.
        # The street's parked cars sit on the road's centre line every 16 m; push them to
        # the far kerb so there is a lane to drive down (x 3.5-7.5), start the player on
        # the near kerb with the car on the road beside him.
        for v in self.s.of_kind("vehicle"):
            v.position = [9.6, 0.0, float(v.position[2])]
        self.h.position = [4.0, 0.0, 0.0]          # 2.4 m from the car: E works at once
        self.car = self.s.add(Entity("car", "vehicle", [5.4, 0.0, 2.0], heading_deg=0.0, size=[1.9, 1.5, 4.4]))
        self.car.attrs["arrived"] = 1.0
        # People crossing the road between the kerbs at various points down the block,
        # and a few walking along it.
        rng = np.random.default_rng(seed + 100)
        for i in range(N_PEDS):
            z = float(rng.uniform(10.0, 85.0))
            if i % 3 == 2:                                     # along the kerb
                x = float(rng.choice([1.5, 2.8]))
                a, b = [x, 0.0, z], [x, 0.0, z + float(rng.uniform(20.0, 40.0)) * float(rng.choice([-1, 1]))]
            else:                                              # across the road
                a, b = [-1.0, 0.0, z], [6.8, 0.0, z + float(rng.uniform(-3.0, 3.0))]   # 6.8: short of the parked cars
            start = a if i % 2 else b
            e = self.s.add(Entity(f"walker{i}", "human", list(start), size=[0.55, 1.75, 0.40]))
            e.attrs["waypoints"] = [b, a] if i % 2 else [a, b]
            e.attrs["wp_radius"] = 1.0
            e.speed = float(rng.uniform(1.0, 1.5))
            t = e.attrs["waypoints"][0]
            e.heading_deg = float(np.degrees(np.arctan2(t[0] - start[0], t[2] - start[2])))
        self.in_car = False
        self.car_v = 0.0
        self.prev_keys: set[str] = set()
        self.ctl = camera(self.spec)
        self.cache = wc.PrimitiveCache()
        self.pitch_trim = 0.0
        # The direction the player is looking, level. WASD are relative to THIS, and the
        # boom anchors on it. It is its own state: arrows turn it, and while the subject
        # walks it eases round behind him. Deriving it from the lens yaw (which carries the
        # shoulder-offset aim) or from the subject's heading (which WASD just changed) each
        # made every W a spiral -- the second one by exactly 45° per frame.
        self.view_az = float(self.h.heading_deg)
        self.vy = 0.0                      # vertical velocity, +up
        self.cam = self.ctl.update(self.h.pos, self.h.heading_deg, DT)
        self.frame = 0
        self.log: list[dict] = []

    def step(self, keys: set[str]) -> Camera:
        h, ctl, car = self.h, self.ctl, self.car
        pressed = keys - self.prev_keys           # key-down edges, for toggles
        self.prev_keys = set(keys)
        if "E" in pressed:
            gap = float(np.hypot(*(car.pos - h.pos)[[0, 2]]))
            if self.in_car:
                self.in_car = False
                h.attrs["riding"] = ""
                th = np.radians(car.heading_deg)
                right = np.array([np.cos(th), 0.0, -np.sin(th)])
                h.pos = car.pos + right * 1.6          # step out on the right
                h.pos = [float(h.pos[0]), 0.0, float(h.pos[2])]
                h.heading_deg = car.heading_deg
                self.view_az = float(car.heading_deg)
                self.s.log("exit_car", "courier")
            elif gap <= 2.5:
                self.in_car = True
                h.attrs["riding"] = "car"
                self.car_v = 0.0
                self.view_az = float(car.heading_deg)
                self.s.log("enter_car", "courier", distance=round(gap, 2))
        if self.in_car:
            return self._drive(keys)
        # --- subject: WASD relative to the lens the player is looking through ------------
        x = (1.0 if "D" in keys else 0.0) - (1.0 if "A" in keys else 0.0)
        z = (1.0 if "W" in keys else 0.0) - (1.0 if "S" in keys else 0.0)
        airborne = h.pos[1] < -1e-6 or self.vy > 0.0
        # --- camera: arrows turn the view, R recentres it ---------------------------------
        if "LEFT" in keys:
            self.view_az -= ORBIT_RATE * DT
        if "RIGHT" in keys:
            self.view_az += ORBIT_RATE * DT
        if "UP" in keys:
            self.pitch_trim = min(25.0, self.pitch_trim + TILT_RATE * DT)
        if "DOWN" in keys:
            self.pitch_trim = max(-25.0, self.pitch_trim - TILT_RATE * DT)
        if "R" in keys:
            self.view_az, self.pitch_trim = float(h.heading_deg), 0.0
        if (x or z) and not airborne:
            h.heading_deg = float(self.view_az + np.degrees(np.arctan2(x, z)))
            h.speed = RUN if "SHIFT" in keys else WALK
            # Ease the view round behind him, as a third-person camera does; holding W+D
            # therefore walks a gentle arc (45° per RECENTRE_TAU), not a straight diagonal.
            d = ((h.heading_deg - self.view_az + 180.0) % 360.0) - 180.0
            self.view_az += d * (1.0 - np.exp(-DT / RECENTRE_TAU))
        elif not airborne:
            h.speed = 0.0
        if "SPACE" in keys and not airborne:
            self.vy = JUMP_V0
        if "F" in keys:
            # Hit: the nearest standing person within 1.2 m and roughly in front goes down.
            fwd = h.forward()
            best = None
            for e in self.s.of_kind("human"):
                if e.eid == "courier" or e.attrs.get("hit"):
                    continue
                d = e.pos - h.pos
                dist = float(np.hypot(d[0], d[2]))
                if dist <= 1.2 and float(d @ fwd) > 0.0 and (best is None or dist < best[0]):
                    best = (dist, e)
            if best is not None:
                best[1].attrs["hit"] = 1.0
                self.s.log("hit", best[1].eid, by="courier", distance=round(best[0], 2))
        # The boom anchors on the view azimuth: the rig reads heading + _orbit.
        ctl._orbit = ((self.view_az - h.heading_deg + 180.0) % 360.0) - 180.0
        # --- world ------------------------------------------------------------------------
        rules.step(self.s, ["waypoints", "separation", "impact", "falling", "blocking", "locomotion"])
        if airborne or self.vy > 0.0:
            p = h.pos
            self.vy -= G * DT
            alt = max(0.0, -p[1] + self.vy * DT)
            p[1] = -alt
            h.pos = p
            if alt <= 0.0:
                self.vy = 0.0
        wc.advance_gait(self.s)
        cam = ctl.update(h.pos, h.heading_deg, DT)
        cam = Camera(focal_px=cam.focal_px, position=cam.position, roll_deg=cam.roll_deg,
                     yaw_deg=cam.yaw_deg,
                     pitch_deg=float(np.clip(cam.pitch_deg + self.pitch_trim,
                                             ctl.pitch_min, ctl.pitch_max)))
        self.log.append({"f": self.frame, "keys": sorted(keys)})
        self.frame += 1
        self.cam = cam
        return cam

    def _drive(self, keys: set[str]) -> Camera:
        """Throttle/brake/steer; the follow rig rides on the driver with a longer boom."""
        h, ctl, car = self.h, self.ctl, self.car
        gas = CAR_ACCEL * (1.6 if "SHIFT" in keys else 1.0)
        v = self.car_v
        if "W" in keys:
            v = min(CAR_VMAX, v + gas * DT)
        elif "S" in keys:
            v = max(CAR_REV, v - CAR_BRAKE * DT) if v > 0.05 else max(CAR_REV, v - 2.0 * DT)
        else:
            v = v - np.sign(v) * min(abs(v), CAR_DRAG * DT)
        steer = (1.0 if "D" in keys else 0.0) - (1.0 if "A" in keys else 0.0)
        if steer and abs(v) > 0.2:
            car.heading_deg += steer * STEER_DEG_S * DT * min(1.0, abs(v) / 4.0) * np.sign(v)
        car.speed = v
        if "LEFT" in keys:
            self.view_az -= ORBIT_RATE * DT
        if "RIGHT" in keys:
            self.view_az += ORBIT_RATE * DT
        if "UP" in keys:
            self.pitch_trim = min(25.0, self.pitch_trim + TILT_RATE * DT)
        if "DOWN" in keys:
            self.pitch_trim = max(-25.0, self.pitch_trim - TILT_RATE * DT)
        if "R" in keys:
            self.view_az, self.pitch_trim = float(car.heading_deg), 0.0
        d = ((car.heading_deg - self.view_az + 180.0) % 360.0) - 180.0
        self.view_az += d * (1.0 - np.exp(-DT / RECENTRE_TAU))
        rules.step(self.s, ["waypoints", "separation", "impact", "falling", "blocking", "locomotion"])
        self.car_v = float(car.speed)          # blocking may have zeroed it
        cp = car.pos
        h.pos = [float(cp[0]), -car.size[1] * 0.55, float(cp[2])]
        h.heading_deg, h.speed = float(car.heading_deg), 0.0
        wc.advance_gait(self.s)
        ctl._orbit = ((self.view_az - h.heading_deg + 180.0) % 360.0) - 180.0
        ctl.distance = 5.0
        cam = ctl.update(h.pos, h.heading_deg, DT)
        ctl.distance = self.spec.boom_len
        cam = Camera(focal_px=cam.focal_px, position=cam.position, roll_deg=cam.roll_deg, yaw_deg=cam.yaw_deg,
                     pitch_deg=float(np.clip(cam.pitch_deg + self.pitch_trim, ctl.pitch_min, ctl.pitch_max)))
        self.log.append({"f": self.frame, "keys": sorted(keys)})
        self.frame += 1
        self.cam = cam
        return cam

    def downed(self) -> int:
        return sum(1 for e in self.s.of_kind("human") if e.attrs.get("hit"))

    def prims(self):
        return self.cache.build(self.s)


# ------------------------------------------------------------------------------ HUD ----
def _record(prev, cam, h, k):
    rec = dict(k=k, pos=cam.position.copy(), pitch=cam.pitch_deg, yaw=cam.yaw_deg,
               roll=cam.roll_deg, hpos=h.pos.copy(), hhead=float(h.heading_deg),
               dist=float(np.linalg.norm(cam.position - h.pos)))
    if prev is None:
        rec.update(v=np.zeros(3), yaw_rate=0.0, pitch_rate=0.0, roll_rate=0.0, turn=0.0,
                   climb=0.0, hspeed=0.0, hv_cam=np.zeros(3))
        return rec
    r_, u_, f_ = moves.lens_axes(cam)
    vw = (rec["pos"] - prev["pos"]) * FPS
    rec["v"] = np.array([vw @ r_, -(vw @ u_), vw @ f_])
    rec["yaw_rate"] = (((cam.yaw_deg - prev["yaw"] + 180) % 360) - 180) * FPS
    rec["pitch_rate"] = (cam.pitch_deg - prev["pitch"]) * FPS
    rec["roll_rate"] = 0.0
    rec["turn"] = (((rec["hhead"] - prev["hhead"] + 180) % 360) - 180) * FPS
    d = rec["hpos"] - prev["hpos"]
    rec["climb"] = -d[1] * FPS
    rec["hspeed"] = float(np.hypot(d[0], d[2])) * FPS
    rec["hv_cam"] = np.array([d @ r_, -(d @ u_), d @ f_]) * FPS
    return rec


def compose(z, ids, rec, keys, fps_now, extra=""):
    """Semantic over depth over the HUD, at 2x, as one RGB array."""
    m = (ids == 8) & (z > 0)
    zsub = float(np.median(z[m])) if m.sum() > 20 else float("nan")
    cmd = "held: " + (" ".join(sorted(keys)) or "-") + f"    {fps_now:4.1f} fps" + extra
    return np.concatenate([moves.up(moves.LUT[ids]), moves.up(moves.depth_img(z)),
                           moves.hud(rec, "live", cmd, zsub)], 0)


# ----------------------------------------------------------------------------- live ----
def play(out_dir: Path, seed: int, script: str | None, max_frames: int | None, video: bool = False):
    import pygame
    pygame.init()
    W, H = 672, 384 + 384 + 150
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("cwm-proxy live  -  WASD move, arrows look, Space jump, R recentre, Esc quit")
    clock = pygame.time.Clock()
    ses = Session(seed)
    keymap = {pygame.K_w: "W", pygame.K_a: "A", pygame.K_s: "S", pygame.K_d: "D",
              pygame.K_LSHIFT: "SHIFT", pygame.K_RSHIFT: "SHIFT", pygame.K_LEFT: "LEFT",
              pygame.K_RIGHT: "RIGHT", pygame.K_UP: "UP", pygame.K_DOWN: "DOWN",
              pygame.K_SPACE: "SPACE", pygame.K_r: "R", pygame.K_f: "F", pygame.K_e: "E"}
    scripted = _parse_script(script) if script else None
    prev, t0, n = None, time.time(), 0
    frames_out = []
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                running = False
        if scripted is not None:
            keys = scripted.get(ses.frame, set())
        else:
            pressed = pygame.key.get_pressed()
            keys = {name for k, name in keymap.items() if pressed[k]}
        cam = ses.step(keys)
        z, ids = render(cam, ses.prims(), width=336, height=192)
        rec = _record(prev, cam, ses.h, ses.frame - 1)
        prev = rec
        n += 1
        fps_now = n / max(time.time() - t0, 1e-6)
        extra = (f"    IN CAR {ses.car_v:4.1f} m/s" if ses.in_car else "    on foot (E near the car to drive)") + f"    downed {ses.downed()}"
        img = compose(z, ids, rec, keys, fps_now, extra)
        if video:
            frames_out.append(img)
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)
        if max_frames is not None and ses.frame >= max_frames:
            running = False
        if scripted is not None and ses.frame >= len(scripted):
            running = False              # a script ends the session; no window to close
    pygame.quit()
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": seed, "fps": FPS, "frames": ses.frame, "inputs": ses.log},
              open(out_dir / "inputs.json", "w"))
    from PIL import Image
    Image.fromarray(img).save(out_dir / "last_frame.png")
    if video:
        moves.write(frames_out, out_dir / "session.mp4")
    print(f"{ses.frame} frames, {fps_now:.1f} fps live -> {out_dir}/inputs.json")
    return ses


def _parse_script(script: str) -> dict[int, set[str]]:
    """'W:40,D:20,LEFT:15,SPACE:1,S:20' -> consecutive spans of held keys, for testing
    without a window. '+' holds several at once: 'W+LEFT:30'."""
    out, f = {}, 0
    for span in script.split(","):
        names, n = span.split(":")
        for _ in range(int(n)):
            out[f] = set(names.split("+"))
            f += 1
    return out


# --------------------------------------------------------------------------- replay ----
def replay(inputs: Path, out: Path):
    """Re-simulate the recorded keys and write published-format conditions."""
    rec = json.load(open(inputs))
    ses = Session(rec["seed"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for entry in rec["inputs"]:
        cam = ses.step(set(entry["keys"]))
        z, ids = render_frame(cam, ses.prims())
        writer.write_frame(out, entry["f"], z, ids)
    print(f"replayed {len(rec['inputs'])} frames in {time.time()-t0:.1f}s -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None, help="session dir (default out/play/<stamp>)")
    ap.add_argument("--replay", type=Path, default=None, help="inputs.json to re-render at full quality")
    ap.add_argument("--script", default=None, help="scripted keys for a headless smoke test, e.g. 'W:40,LEFT:15'")
    ap.add_argument("--frames", type=int, default=None, help="stop after this many frames")
    ap.add_argument("--video", action="store_true", help="also save the live window as session.mp4")
    a = ap.parse_args()
    if a.replay:
        replay(a.replay, a.out or a.replay.parent / "conditions")
        return 0
    out = a.out or Path("out/play") / time.strftime("%Y%m%d_%H%M%S")
    play(out, a.seed, a.script, a.frames, a.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
