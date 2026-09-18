"""Unproject -> per-class 3D connected components -> best-of-N primitive fit.

CAVEAT that every residual below must be read with: a single depth frame gives
only the VISIBLE SURFACE of an object.  A primitive fit to a visible surface is
under-constrained (the far side is free), so these residuals are a LOWER BOUND
on true shape error.  A shape that fits badly here fits at least this badly in
full 3D; a shape that fits well here may still be wrong behind.
"""
import numpy as np
from scipy import ndimage
H,W=192,336; CX,CY=W/2-0.5,H/2-0.5
_J,_I=np.mgrid[0:H,0:W]

def unproject(z,f): return np.stack([z*(_I-CX)/f, z*(_J-CY)/f, z],-1)

def components(pts, mask, vox=0.30, minpts=120):
    """3D connected components on a voxel grid."""
    p=pts[mask]; idx=np.flatnonzero(mask)
    if len(p)<minpts: return []
    lo=p.min(0); g=np.floor((p-lo)/vox).astype(np.int64)
    shape=g.max(0)+3
    if np.prod(shape)>6e7: vox*=2; g=np.floor((p-lo)/vox).astype(np.int64); shape=g.max(0)+3
    occ=np.zeros(shape,bool); occ[g[:,0],g[:,1],g[:,2]]=True
    lab,nl=ndimage.label(occ, structure=np.ones((3,3,3)))
    li=lab[g[:,0],g[:,1],g[:,2]]
    out=[]
    for k in range(1,nl+1):
        sel=li==k
        if sel.sum()<minpts: continue
        out.append((p[sel], idx[sel]))
    out.sort(key=lambda a:-len(a[0]))
    return out

# ---- primitive fits: each returns (name, residual_metres_rms, params, extra)
def fit_plane(p):
    c=p.mean(0); u,s,vt=np.linalg.svd(p-c,full_matrices=False)
    n=vt[2]; r=(p-c)@n
    return ("Plane", float(np.sqrt((r**2).mean())), dict(point=c.tolist(), normal=n.tolist()))

def fit_rect(p):
    """Horizontal quad: plane of constant height along the scene's up axis (axis 1, +Y down)."""
    y=p[:,1]; r=y-np.median(y)
    return ("Rect", float(np.sqrt((r**2).mean())),
            dict(y=float(np.median(y)), x=[float(p[:,0].min()),float(p[:,0].max())],
                 z=[float(p[:,2].min()),float(p[:,2].max())]))

def fit_sphere(p):
    A=np.concatenate([2*p, np.ones((len(p),1))],1); b=(p**2).sum(1)
    sol,*_=np.linalg.lstsq(A,b,rcond=None); c=sol[:3]; rad=np.sqrt(max(sol[3]+(c**2).sum(),1e-9))
    d=np.linalg.norm(p-c,axis=1)-rad
    return ("Sphere", float(np.sqrt((d**2).mean())), dict(centre=c.tolist(), radius=float(rad)))

def fit_capsule(p):
    c=p.mean(0); u,s,vt=np.linalg.svd(p-c,full_matrices=False); ax=vt[0]
    t=(p-c)@ax; a=c+ax*t.min(); b=c+ax*t.max()
    ba=b-a; L2=ba@ba
    q=np.clip(((p-a)@ba)/L2,0,1)
    d=np.linalg.norm(p-(a+q[:,None]*ba),axis=1)
    rad=float(np.median(d))
    return ("Capsule", float(np.sqrt(((d-rad)**2).mean())),
            dict(a=a.tolist(), b=b.tolist(), radius=rad))

def fit_box(p):
    c=p.mean(0); u,s,vt=np.linalg.svd(p-c,full_matrices=False); R=vt
    q=(p-c)@R.T; half=np.abs(q).max(0)
    # distance from point to box SURFACE (points are on the visible hull)
    inside=np.all(np.abs(q)<=half,1)
    dsurf=np.min(half-np.abs(q),1)      # distance to nearest face, for interior pts
    d=np.where(inside, dsurf, 0.0)
    return ("Box", float(np.sqrt((d**2).mean())),
            dict(centre=c.tolist(), half=half.tolist(), axes=R.tolist()))

def fit_sphere_cluster(p, k=12, iters=25, seed=0):
    """k-means lobes, each lobe a sphere; residual to nearest lobe surface."""
    rng=np.random.default_rng(seed)
    C=p[rng.choice(len(p),min(k,len(p)),replace=False)].copy()
    for _ in range(iters):
        d=((p[:,None,:]-C[None])**2).sum(-1); a=d.argmin(1)
        for j in range(len(C)):
            m=a==j
            if m.sum()>3: C[j]=p[m].mean(0)
    d=np.sqrt(((p[:,None,:]-C[None])**2).sum(-1)); a=d.argmin(1)
    rad=np.array([np.median(d[a==j,j]) if (a==j).sum()>3 else 0.0 for j in range(len(C))])
    e=d[np.arange(len(p)),a]-rad[a]
    return ("SphereCluster", float(np.sqrt((e**2).mean())),
            dict(n=int(len(C)), radii=[float(x) for x in rad]))

def best_fit(p, try_cluster=True):
    cands=[fit_plane(p), fit_rect(p), fit_sphere(p), fit_capsule(p), fit_box(p)]
    if try_cluster and len(p)>=60: cands.append(fit_sphere_cluster(p))
    cands.sort(key=lambda c:c[1])
    return cands
