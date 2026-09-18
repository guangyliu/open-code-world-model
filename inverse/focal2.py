"""Focal from two-frame rigidity, global search (no initialisation chain).

For every candidate f we run ICP *entirely in that f's own geometry* and score
how rigid the pair looks.  A wrong f applies S=diag(f0/f,f0/f,1) to the true
cloud; S R S^-1 is a rotation only when R commutes with S, so camera rotation
(not translation) is what makes f observable.
"""
import numpy as np
from scipy.spatial import cKDTree
H, W = 192, 336
CX, CY = W/2-0.5, H/2-0.5
_J, _I = np.mgrid[0:H, 0:W]
STATIC = (0, 2, 4, 5)          # terrain/water/road/vegetation; 8,9,10 move

def unproject(z, f):
    return np.stack([z*(_I-CX)/f, z*(_J-CY)/f, z], -1)

def normals(pts, valid):
    dx = np.zeros_like(pts); dy = np.zeros_like(pts)
    dx[:,1:-1] = pts[:,2:]-pts[:,:-2]; dy[1:-1,:] = pts[2:,:]-pts[:-2,:]
    n = np.cross(dx, dy); ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return n/np.where(ln>1e-9, ln, 1.0), (ln[...,0]>1e-9)&valid

def _noedge(z, rel=0.04):
    bad = np.zeros(z.shape, bool)
    zc = np.where(z>0, z, np.nan)
    for di,dj in ((0,1),(0,-1),(1,0),(-1,0)):
        r = np.roll(np.roll(zc,di,0),dj,1)
        with np.errstate(invalid="ignore"):
            bad |= ~(np.abs(r-zc) <= rel*zc)
    return ~bad

def mask(z, s, zmax, stride, static_only=True):
    m = (z>0)&(z<zmax)&_noedge(z)
    if static_only: m &= np.isin(s, STATIC)
    k = np.zeros_like(m); k[::stride,::stride] = True
    return m & k

def icp(z0, s0, z1, s1, f, zmax, stride=2, iters=40):
    p0 = unproject(z0,f); p1 = unproject(z1,f)
    n0, okn = normals(p0, z0>0)
    m0 = mask(z0,s0,zmax,stride)&okn; m1 = mask(z1,s1,zmax,stride)
    if m0.sum()<400 or m1.sum()<400: return None
    Q, N, P = p0[m0], n0[m0], p1[m1]
    tree = cKDTree(Q); R = np.eye(3); t = np.zeros(3)
    res = np.inf
    for it in range(iters):
        Pt = P@R.T+t
        d, j = tree.query(Pt, k=1)
        thr = max(3.0*np.median(d), 0.01)
        keep = d < thr
        if keep.sum() < 200: return None
        A, B, nb = Pt[keep], Q[j[keep]], N[j[keep]]
        M = np.concatenate([np.cross(A,nb), nb],1)
        b = np.einsum("ij,ij->i",(B-A),nb)
        sol,*_ = np.linalg.lstsq(M,b,rcond=None)
        a, tt = sol[:3], sol[3:]
        K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
        u,_,vt = np.linalg.svd(np.eye(3)+K); dR = u@vt
        R = dR@R; t = dR@t+tt
        res = np.median(np.abs(np.einsum("ij,ij->i",(B-A),nb)))
    rad = np.sqrt(((Q-Q.mean(0))**2).sum(1).mean())
    ang = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
    return dict(res=res, rad=rad, score=res/rad, n=int(keep.sum()), ang=ang,
                trans=float(np.linalg.norm(t)), R=R, t=t)

def curve(z0,s0,z1,s1,focals,zmax,stride=2):
    out=[]
    for f in focals:
        r = icp(z0,s0,z1,s1,float(f),zmax,stride)
        out.append((f, np.nan, np.nan, np.nan, np.nan) if r is None
                   else (f, r["res"], r["rad"], r["score"], r["ang"]))
    return np.array(out)

def argmin_parabolic(tab, col=3):
    v = tab[:,col]; k = int(np.nanargmin(v))
    if 0 < k < len(v)-1 and np.all(np.isfinite(v[k-1:k+2])):
        y0,y1,y2 = v[k-1:k+2]; den = y0-2*y1+y2
        if den > 0:
            d = 0.5*(y0-y2)/den
            return float(tab[k,0] + d*(tab[k+1,0]-tab[k,0])), float(v[k])
    return float(tab[k,0]), float(v[k])
