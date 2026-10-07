"""Photo-anchored clay proxies for LTX-2.5 Layout-To-Render: lift a still into geometry,
put the capsule rig where the person stands, then animate the rig or the camera.

    python examples/photo_depth.py /path/to/img out/photo      # GPU: depth, focal, masks
    python examples/photo_clips.py out/photo [stem ...]         # CPU: clay clips

Frame 0 of every clip is the photo itself, so the first frame the LoRA is given and the
clay it follows agree by construction -- the gallery scenes could only approximate that.

Geometry:
  * the photo's pixels are unprojected with Depth Pro's metric depth and focal length into
    a point cloud with normals, rendered by z-buffered splatting;
  * roll is 0; pitch comes from a floor fit beside the feet when enough floor is visible,
    else an annotated horizon row, else level;
  * the person is cut out of the cloud and replaced by the shared capsule rig, rigid --
    fixed pose, no gait, no idle sway -- standing under the head of the mask at the body's
    depth, as tall as head-to-boots when the feet are in frame and 1.7 m when they are not.
    The ground plane (y = 0) passes through the feet;
  * the hole the person leaves is filled with the ground plane where a ray reaches it,
    and with inpainted background depth above the horizon, so a rig that walks away
    uncovers plausible geometry instead of a silhouette-shaped void;
  * a 6 m ground disc around the camera stands in for floor the photo never saw.

Clips: four rig actions with the camera locked, four camera moves with the rig still.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage, optimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compiler import rig                                       # noqa: E402
from compiler.render import OUT_W, Camera, Plane, clay_surface, shade_clay   # noqa: E402

FPS, FRAMES = 24, 121          # 8k+1, the length the pipeline takes
OUT_H = 512                     # 480p-class; width follows the photo, rounded to /64
SS = 2                          # clay supersampling

# Short first-frame descriptions for the prompt (the LoRA card: one or two sentences that
# match the first frame; never "clay" or "3D").
PHOTOS = {
    "00ad66ded066_0": dict(pron="He", desc=(
        "A blond young man in a dark hooded jacket with a glowing blue gauntlet stands on a "
        "wet rooftop at night, overlooking a neon cyberpunk city of towering skyscrapers and "
        "giant holographic billboards.")),
    "23daf2feaefa_0": dict(pron="She", horizon=0.28, desc=(
        "A slender young woman with long blue braids, purple shorts and black boots, holding a "
        "large shark-mouthed rocket launcher, stands on a grassy rocky clifftop above a "
        "sparkling blue sea, with a coastal city of towers on the cliffs to the right under a "
        "bright sky.")),
    "8e1891bc6ff1_0": dict(pron="He", desc=(
        "The Monkey King in ornate red and gold armour, a golden staff over his shoulder, "
        "stands in a vast stone-paved palace courtyard facing a grand imperial hall with golden "
        "roofs, white marble terraces and bronze cauldrons under a clear blue sky.")),
    "c89416a7293a_0": dict(pron="She", desc=(
        "A young woman with long blue braids and tattoos, holding a large graffiti-covered gun, "
        "stands on a dark rooftop at dusk overlooking a neon cyberpunk city with giant "
        "billboards, a glowing river and flying vehicles.")),
}


# --------------------------------------------------------------------------- lifting

def _ry(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _ransac_plane(P, iters=400, tol=0.03, seed=0):
    """Plane through the most points, tolerance relative to depth. Returns (n, d), n.P + d = 0."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    rel = tol * np.linalg.norm(P, axis=1)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        inl = np.abs(P @ n + d) < rel
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    return n, -n @ c, best_n / len(P)


class PhotoScene:
    def __init__(self, d: Path):
        g = np.load(d / "geom.npz")
        img = Image.open(d / "image.png").convert("RGB")
        Z, f = g["depth"].astype(np.float64), float(g["focal"])
        lab = g["labels"]
        h0, w0 = Z.shape
        # Output size: height 512, width the photo's aspect rounded to a multiple of 64; the
        # photo is centre-cropped to that aspect so the principal point stays centred.
        self.H = OUT_H
        self.W = int(round(OUT_H * w0 / h0 / 64)) * 64
        asp = self.W / self.H
        cw, ch = (w0, int(round(w0 / asp))) if w0 / h0 < asp else (int(round(h0 * asp)), h0)
        y0, x0 = (h0 - ch) // 2, (w0 - cw) // 2
        sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
        Z, lab = Z[sl], lab[sl]
        self.first = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((self.W, self.H), Image.LANCZOS)
        self.f_out = f * self.W / cw

        person = ndimage.binary_dilation(lab == g["person"], iterations=4)
        sky = (lab == g["sky"]) | (Z > 400)
        v, u = np.mgrid[0:ch, 0:cw]
        ray = np.stack([(u + 0.5 - cw / 2) / f, (v + 0.5 - ch / 2) / f, np.ones_like(Z)], -1)

        # Up and pitch. Roll is taken as 0: every subject here stands upright in frame. Pitch
        # comes from a floor fit beside the feet when there is enough floor to trust, else from
        # an annotated horizon row, else level. A fit over the whole bottom band was tried
        # first and failed on all four photos -- rock slopes, clutter, and floors cut off by
        # the frame gave camera heights of 3-30 cm and up to 21 degrees of pitch.
        pm = lab == g["person"]
        ys, xs = np.nonzero(pm)
        top, bot = ys.min(), ys.max()
        zp = float(np.median(Z[ndimage.binary_erosion(pm, iterations=6)]))
        P = ray * Z[..., None]
        xl, xr = np.percentile(xs, [5, 95])
        floor = (~person & ~sky & (v > bot - 0.12 * ch) & (np.abs(Z - zp) < 0.6 * zp)
                 & (np.abs(u - (xl + xr) / 2) < (xr - xl) / 2 + 0.15 * cw))
        meta = PHOTOS.get(d.name, {})
        pitch, how = 0.0, "level"
        if floor.sum() > 30000:
            n, _, frac = _ransac_plane(P[floor][:: max(1, floor.sum() // 20000)])
            n = -n if n[1] > 0 else n
            if frac > 0.8:
                pitch, how = float(np.clip(np.degrees(np.arcsin(-n[2])), -20, 10)), "floor"
        if how == "level" and "horizon" in meta:
            pitch, how = -float(np.degrees(np.arctan((ch / 2 - meta["horizon"] * h0 + y0) / f))), "horizon"
        self.pitch, self.roll = pitch, 0.0
        R = Camera(pitch_deg=pitch).basis()             # world -> camera rows
        self.R0 = R

        # Person. Depth at the body, not a ground intersection: the head at the body's depth,
        # the feet at the boots' depth when they are in frame, or 1.7 m below the head when
        # the frame cuts them off.
        hu = float(np.median(xs[ys <= top + 3]))
        head = np.array([(hu + 0.5 - cw / 2) / f, (top + 0.5 - ch / 2) / f, 1.0]) * zp @ R
        hidden = bot >= ch - 3
        if not hidden:
            boots = pm & (v >= bot - 0.04 * (bot - top))
            fz = float(np.median(Z[boots]))
            fu = float(np.median(xs[ys >= bot - 0.04 * (bot - top)]))
            feet = np.array([(fu + 0.5 - cw / 2) / f, (bot + 0.5 - ch / 2) / f, 1.0]) * fz @ R
            height = float(np.clip(feet[1] - head[1], 1.0, 2.2))
        else:
            height = 1.7
        feet = np.array([head[0], head[1] + height, head[2]])
        self.cam_h = float(feet[1])                     # level frame: camera at the origin
        self.cam0 = dict(position=np.array([0.0, -self.cam_h, 0.0]), yaw=0.0, pitch=pitch, roll=0.0)
        o = self.cam0["position"]
        F = feet + o
        F[1] = 0.0
        self.feet, self.height = F, height
        self.heading = float(np.degrees(np.arctan2(F[0] - o[0], F[2] - o[2])))
        self.scale = height / rig.HEIGHT
        self.unit = height / 1.7                        # motion distances scale with the body

        def to_world(Pc):
            return Pc @ R + o

        def ground_hit(rays):
            """Camera z at which each ray meets the ground (y = 0), inf if it never does."""
            dw = rays @ R
            return np.where(dw[..., 1] > 1e-6, self.cam_h / np.maximum(dw[..., 1], 1e-6), np.inf)

        # background: fill the person hole. The hole reaches 10 px past the mask so the
        # flying pixels where body depth met background depth go with it. The fill is a
        # normalised convolution of log depth at three widths (smooth, unlike cv2.inpaint,
        # whose streaks shaded as ribs), replaced by the ground plane wherever a ray reaches
        # the ground in front of it.
        hole = ndimage.binary_dilation(lab == g["person"], iterations=10)
        logz = np.log(np.clip(Z, 0.1, 1000))
        valid = (~hole & ~sky).astype(np.float64)
        fill = np.zeros_like(logz)
        got = np.zeros_like(logz, bool)
        for sig in (12, 36, 100):
            num = ndimage.gaussian_filter(logz * valid, sig)
            den = ndimage.gaussian_filter(valid, sig)
            ok = (den > 0.05) & ~got
            fill = np.where(ok, num / np.maximum(den, 1e-9), fill)
            got |= ok
        Zf = np.where(hole, np.exp(fill), Z)
        gz = ground_hit(ray)
        Zf = np.where(hole & (gz < Zf * 1.5), np.minimum(gz, Zf * 1.5), Zf)
        person = hole
        Zs = np.exp(ndimage.gaussian_filter(np.log(np.clip(Zf, 0.1, 1000)), 1.2))
        P = ray * Zs[..., None]
        du = np.gradient(P, axis=1)
        dv = np.gradient(P, axis=0)
        nrm = np.cross(dv, du)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
        nrm = np.where((np.einsum("...i,...i->...", nrm, P) > 0)[..., None], -nrm, nrm)
        gl = np.hypot(*np.gradient(np.log(np.clip(Zf, 0.1, 1000))))
        keep = ~sky & ((gl < 0.06) | hole)              # flying pixels out, the fill in
        self.pts = to_world((ray * Zf[..., None])[keep])
        self.nrm = nrm[keep] @ R                       # camera -> world axes
        self.info = dict(W=self.W, H=self.H, f_out=round(self.f_out, 1), cam_h=round(self.cam_h, 2),
                         pitch=round(self.pitch, 1), pitch_from=how, person_z=round(zp, 2),
                         feet=[round(float(x), 2) for x in F], height=round(height, 2),
                         heading=round(self.heading, 1), feet_hidden=bool(hidden), points=int(keep.sum()))

    def _cam(self, position, yaw, pitch, roll):
        return Camera(focal_px=getattr(self, "f_out", 1.0) * OUT_W / getattr(self, "W", OUT_W),
                      position=np.asarray(position, float), pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)

    # ----------------------------------------------------------------- rendering
    def render(self, cam: dict, body: dict) -> np.ndarray:
        W, H = self.W * SS, self.H * SS
        c = self._cam(**cam)
        prims = rig.Human(position=np.asarray(body["pos"], float), facing_deg=body["heading"],
                          pose=rig.Pose(), scale=self.scale).primitives()
        prims.append(Plane(np.array([c.position[0], 0.0, c.position[2]]), np.array([0.0, -1.0, 0.0]),
                           class_id=0, radius=6.0))
        dist, nc, hit = clay_surface(c, prims, W, H)

        R = c.basis()
        pc = (self.pts - c.position) @ R.T
        z = pc[:, 2]
        ok = z > 0.3
        f = self.f_out * SS
        u = np.floor(f * pc[ok, 0] / z[ok] + W / 2).astype(np.int64)
        v = np.floor(f * pc[ok, 1] / z[ok] + H / 2).astype(np.int64)
        r = np.linalg.norm(pc[ok], axis=1)
        nn = self.nrm[ok] @ R.T
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, r, nn = u[inb], v[inb], r[inb], nn[inb]
        order = np.argsort(-r)                          # far first; the nearest write wins
        pd = np.full((H, W), np.inf)
        pn = np.zeros((H, W, 3))
        pd[v[order], u[order]] = r[order]
        pn[v[order], u[order]] = nn[order]
        for _ in range(3):                              # close splat gaps from magnification
            empty = ~np.isfinite(pd)
            if not empty.any():
                break
            bd, bn = pd.copy(), pn.copy()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                sd = np.roll(pd, (dy, dx), (0, 1))
                sn = np.roll(pn, (dy, dx), (0, 1))
                take = empty & (sd < bd)
                bd = np.where(take, sd, bd)
                bn = np.where(take[..., None], sn, bn)
            pd, pn = bd, bn
        use = pd < dist
        dist = np.where(use, pd, dist)
        nc = np.where(use[..., None], pn, nc)
        return shade_clay(dist, nc, np.isfinite(dist), supersample=SS)


# --------------------------------------------------------------------------- actions

def _ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _seg(t, a, b):
    return _ease((t - a) / (b - a))


def plan(sc: PhotoScene, action: str):
    """Per-frame (camera, body) for one clip. Rig is rigid: only its transform moves."""
    cam0, F, hd = sc.cam0, sc.feet, sc.heading
    o = cam0["position"]
    d0 = float(np.hypot(*(F - o)[[0, 2]]))
    out = []
    for k in range(FRAMES):
        t = k / FPS
        cam = dict(cam0)
        pos, heading = F.copy(), hd
        if action == "turn_around":
            heading = hd + 180.0 * _seg(t, 0.6, 2.6)
        elif action in ("walk_left", "walk_right"):
            # Diagonally away, 35 degrees off the view axis: straight sideways leaves a
            # close-framed subject's frame within two seconds.
            s = -1.0 if action == "walk_left" else 1.0
            heading = hd + s * 35.0 * _seg(t, 0.0, 0.7)
            dist = 0.6 * sc.unit * max(t - 0.5, 0.0)
            pos = F + dist * np.array([np.sin(np.radians(hd + s * 35)), 0.0, np.cos(np.radians(hd + s * 35))])
        elif action == "jump":
            for t0 in (0.8, 2.6):
                tau = t - t0
                if 0.0 <= tau <= 2 * 3.2 / 9.81:
                    pos[1] = -(3.2 * tau - 0.5 * 9.81 * tau * tau)
        elif action == "dolly_in":
            e = _seg(t, 0.0, FRAMES / FPS)
            cam["position"] = o + 0.30 * d0 * e * np.array([np.sin(np.radians(hd)), 0.0, np.cos(np.radians(hd))])
        elif action == "orbit":
            a = 30.0 * _seg(t, 0.0, FRAMES / FPS)
            cam["position"] = F + _ry(a) @ (o - F)
            cam["yaw"] = cam0["yaw"] + a
        elif action == "crane_up":
            e = _seg(t, 0.0, FRAMES / FPS)
            rise = 0.8 * sc.unit * e
            cam["position"] = o + np.array([0.0, -rise, 0.0])
            mid = 0.55 * sc.height
            look0 = np.degrees(np.arctan2(sc.cam_h - mid, d0))
            look1 = np.degrees(np.arctan2(sc.cam_h + rise - mid, d0))
            cam["pitch"] = cam0["pitch"] - (look1 - look0)
        elif action == "pan_right":
            cam["yaw"] = cam0["yaw"] + 25.0 * _seg(t, 0.0, FRAMES / FPS)
        else:
            raise ValueError(action)
        out.append((cam, dict(pos=pos, heading=heading)))
    return out


ACTIONS = {
    "turn_around": "{P} slowly turns around on the spot to face the camera.",
    "walk_left": "{P} turns to the left and walks steadily away, diagonally across the ground.",
    "walk_right": "{P} turns to the right and walks steadily away, diagonally across the ground.",
    "jump": "{P} jumps straight up twice in place, landing lightly each time.",
    "dolly_in": "{P} stands still as the camera pushes slowly in toward {o} back.",
    "orbit": "{P} stands still as the camera arcs slowly around {o}, revealing more of the view.",
    "crane_up": "{P} stands still as the camera rises smoothly and tilts down over {o} shoulder.",
    "pan_right": "{P} stands still as the camera pans slowly to the right across the scene.",
}


# --------------------------------------------------------------------------- driver

_SC = None


def _frame(job):
    k, cam, body = job
    return k, _SC.render(cam, body)


def write_mp4(frames, path, W, H):
    c = av.open(str(path), mode="w")
    s = c.add_stream("libx264", rate=FPS)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.options = {"crf": "12", "preset": "slow"}
    for fr in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def run(d: Path, actions) -> None:
    global _SC
    t0 = time.time()
    _SC = sc = PhotoScene(d)
    print(f"{d.name}: {sc.info}  ({time.time() - t0:.1f}s)")
    meta = PHOTOS.get(d.name, dict(pron="They", desc="A person stands in the scene."))
    pron = meta["pron"]
    poss = {"He": "his", "She": "her"}.get(pron, "their")
    sc.first.save(d / "first_frame.png")
    with mp.get_context("fork").Pool(int(os.environ.get("CWM_WORKERS", 32))) as pool:
        for a in actions:
            t1 = time.time()
            o = d / "clips" / a
            o.mkdir(parents=True, exist_ok=True)
            frames = dict(pool.imap_unordered(_frame, [(k, c, b) for k, (c, b) in enumerate(plan(sc, a))]))
            frames = [frames[k] for k in range(FRAMES)]
            write_mp4(frames, o / "clay.mp4", sc.W, sc.H)
            sc.first.save(o / "first_frame.png")
            Image.fromarray(frames[0]).save(o / "clay_000.png")
            text = meta["desc"] + " " + ACTIONS[a].format(P=pron, o=poss)
            (o / "prompt.txt").write_text(text + "\n", encoding="utf-8")
            print(f"  {a:12s} {FRAMES} frames {sc.W}x{sc.H} in {time.time() - t1:.1f}s")


def main() -> int:
    root = Path(sys.argv[1])
    stems = sys.argv[2:] or sorted(p.name for p in root.iterdir() if (p / "geom.npz").exists())
    acts = os.environ.get("CWM_ACTIONS", ",".join(ACTIONS)).split(",")
    for s in stems:
        run(root / s, acts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
