"""Focal recovery from two depth frames of a static scene.

Key fact this is built around: a ground plane CANNOT determine focal length.
For any pinhole f, 1/z of a planar surface is affine in PIXEL coordinates, so
plane-fitting is exactly f-invariant.  What breaks the degeneracy is camera
ROTATION between two frames of a rigid scene: unprojecting with a wrong focal
applies the anisotropic map S = diag(f0/f, f0/f, 1) to the true cloud, and
S R S^-1 is not a rotation unless R commutes with S (i.e. pure translation, or
rotation about the optical axis).  So we measure how non-rigid the frame pair
looks as a function of f.

Pixel correspondences are f-independent, so we: (1) find correspondences by ICP
at a reference focal, (2) sweep f with those correspondences fixed using Kabsch,
(3) re-run (1) at the winner and iterate.
"""
import numpy as np
from scipy.spatial import cKDTree

H, W = 192, 336
CX, CY = W/2 - 0.5, H/2 - 0.5
_J, _I = np.mgrid[0:H, 0:W]

def unproject(z, f):
    """Camera-frame points from z-depth (metres along optical axis)."""
    return np.stack([z*(_I-CX)/f, z*(_J-CY)/f, z], -1)

def depth_normals(pts, valid):
    """Per-pixel normals from the depth grid via neighbour cross products."""
    dx = np.zeros_like(pts); dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[1:-1, :] = pts[2:, :] - pts[:-2, :]
    n = np.cross(dx, dy)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = (ln[..., 0] > 1e-9) & valid
    return n/np.where(ln > 1e-9, ln, 1.0), ok

def _edge_mask(z, rel=0.05):
    """Drop pixels next to a depth discontinuity: the 2x2 min-pool makes them lie."""
    zc = np.where(z > 0, z, np.nan)
    m = np.zeros_like(z, bool)
    for di, dj in ((0,1),(0,-1),(1,0),(-1,0)):
        r = np.roll(np.roll(zc, di, 0), dj, 1)
        with np.errstate(invalid="ignore"):
            m |= np.abs(r-zc) > rel*zc
    return ~np.nan_to_num(m, nan=True).astype(bool) if False else ~(m | ~np.isfinite(zc))

def sample(z, sem, zmax, stride=2):
    v = (z > 0) & (sem != 1) & (z < zmax) & _edge_mask(z)
    v[::stride] = v[::stride]
    keep = np.zeros_like(v); keep[::stride, ::stride] = True
    return v & keep

def kabsch(Pp, Qq, w=None):
    w = np.ones(len(Pp)) if w is None else w
    w = w/w.sum()
    pc = (Pp*w[:,None]).sum(0); qc = (Qq*w[:,None]).sum(0)
    Hm = ((Pp-pc)*w[:,None]).T @ (Qq-qc)
    U,S,Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T@U.T))
    R = Vt.T @ np.diag([1,1,d]) @ U.T
    return R, qc - R@pc

def icp_corr(z0, s0, z1, s1, f, zmax, iters=40, stride=2):
    """Pixel-index correspondences frame1 -> frame0, found at focal f."""
    p0 = unproject(z0, f); p1 = unproject(z1, f)
    n0, okn = depth_normals(p0, z0 > 0)
    m0 = sample(z0, s0, zmax, stride) & okn
    m1 = sample(z1, s1, zmax, stride)
    Q = p0[m0]; N = n0[m0]; idx0 = np.flatnonzero(m0)
    P = p1[m1]; idx1 = np.flatnonzero(m1)
    tree = cKDTree(Q)
    R = np.eye(3); t = np.zeros(3)
    pair = None
    for it in range(iters):
        Pt = P@R.T + t
        d, j = tree.query(Pt, k=1)
        thr = max(3*np.median(d), 0.02)
        keep = d < thr
        if keep.sum() < 200: break
        A = Pt[keep]; B = Q[j[keep]]; nb = N[j[keep]]
        # linearized point-to-plane
        c = np.cross(A, nb)
        M = np.concatenate([c, nb], 1)
        b = np.einsum("ij,ij->i", (B-A), nb)
        sol, *_ = np.linalg.lstsq(M, b, rcond=None)
        a = sol[:3]; tt = sol[3:]
        K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
        dR = np.eye(3)+K
        u,_,vt = np.linalg.svd(dR); dR = u@vt
        R = dR@R; t = dR@t + tt
        pair = (idx1[keep], idx0[j[keep]])
    return pair, R, t

def sweep(z0, z1, pair, focals):
    i1, i0 = pair
    out = []
    for f in focals:
        P = unproject(z1, f).reshape(-1,3)[i1]
        Q = unproject(z0, f).reshape(-1,3)[i0]
        R, t = kabsch(P, Q)
        r = np.linalg.norm(P@R.T + t - Q, axis=1)
        rad = np.sqrt(((Q-Q.mean(0))**2).sum(1).mean())
        out.append((f, np.median(r), np.sqrt((r**2).mean()), rad, np.median(r)/rad))
    return np.array(out)

def recover(z0, s0, z1, s1, focals, zmax=None, f_ref=206.0, rounds=3, stride=2):
    if zmax is None:
        zz = z0[z0>0]; zmax = float(np.percentile(zz, 80))
    f = f_ref; hist=[]
    for k in range(rounds):
        pair, R, t = icp_corr(z0, s0, z1, s1, f, zmax, stride=stride)
        if pair is None or len(pair[0]) < 300: return None, None, None
        tab = sweep(z0, z1, pair, focals)
        fbest = float(tab[np.argmin(tab[:,4]), 0])
        ang = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
        hist.append((f, fbest, len(pair[0]), ang, float(np.linalg.norm(t))))
        f = fbest
    return f, tab, hist
