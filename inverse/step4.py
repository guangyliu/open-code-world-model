import os
"""Fit primitives to each 3D segment, then SCORE BY RE-RENDERING.

Why not score by point-to-surface residual: from one view we see only a surface
sheet, and a plane fits a sheet trivially -- in the raw-residual ranking a human
body and a cart both came out 'best = Plane' at 3 cm RMS.  That is an artifact of
the metric, not a finding about the content.  A primitive is only right if,
rendered from the same camera, it reproduces BOTH the depth and the silhouette.

Everything is done in CAMERA coordinates (renderer at origin, zero rotation), so
there is no pose bookkeeping: world == camera frame.
"""
import sys, numpy as np, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from compiler.render import (Camera, Plane, Rect, Sphere, Capsule, Box, SphereCluster,
                             render_frame, vegetation_clump)
import fitlib as FL
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR)
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
NAME={0:"terrain",1:"sky",2:"water",4:"road",5:"veg",8:"human",9:"animal",10:"vehicle"}

def load(n,fr=0,sh=0):
    d=np.load(f"{P}/{n}/condition/{sh:03d}.npz")
    z=d["depth"][fr].astype(np.float64); s=d["semantic_id"][fr]
    if META[n].startswith("uint16"):
        z=np.where(z>0,NEAR*np.exp(L*(1.0-z/65535.0)),0.0); z=np.where(z>=FAR*0.999,0.0,z)
    return z,s

def cands_for(p, cid):
    """Candidate primitives (camera coords) for one segment's point cloud."""
    out=[]
    c=p.mean(0); u,sv,vt=np.linalg.svd(p-c,full_matrices=False)
    q=(p-c)@vt.T; half=np.abs(q).max(0)
    out.append(("Box", Box(c, half, vt, class_id=cid)))
    # sphere
    A=np.concatenate([2*p,np.ones((len(p),1))],1); b=(p**2).sum(1)
    sol,*_=np.linalg.lstsq(A,b,rcond=None); cc=sol[:3]; rad=np.sqrt(max(sol[3]+(cc**2).sum(),1e-9))
    out.append(("Sphere", Sphere(cc, float(rad), class_id=cid)))
    # ellipsoid (Sphere with per-axis scale, axis-aligned to camera axes)
    ex=(p.max(0)-p.min(0))/2.0; ctr=(p.max(0)+p.min(0))/2.0
    sc=np.maximum(ex, 1e-3); out.append(("Ellipsoid", Sphere(ctr, 1.0, class_id=cid, scale=sc)))
    # capsule along principal axis
    ax=vt[0]; t=(p-c)@ax; a0=c+ax*t.min(); b0=c+ax*t.max()
    ba=b0-a0; L2=max(ba@ba,1e-9); qq=np.clip(((p-a0)@ba)/L2,0,1)
    dd=np.linalg.norm(p-(a0+qq[:,None]*ba),axis=1)
    out.append(("Capsule", Capsule(a0,b0,float(np.median(dd)) or 0.05, class_id=cid)))
    # plane (unbounded) and a bounded disc
    nrm=vt[2]; out.append(("Plane", Plane(c, nrm, class_id=cid)))
    rr=float(np.linalg.norm(p-c,axis=1).max())
    out.append(("PlaneDisc", Plane(c, nrm, class_id=cid, radius=rr)))
    # horizontal rect at median height
    out.append(("Rect", Rect(y=float(np.median(p[:,1])), x0=float(p[:,0].min()), x1=float(p[:,0].max()),
                             z0=float(p[:,2].min()), z1=float(p[:,2].max()), class_id=cid)))
    # sphere clusters at several lobe counts
    for k in (6,12,24):
        if len(p)<4*k: continue
        rng=np.random.default_rng(0)
        C=p[rng.choice(len(p),k,replace=False)].copy()
        for _ in range(25):
            d2=((p[:,None,:]-C[None])**2).sum(-1); a_=d2.argmin(1)
            for j in range(k):
                m=a_==j
                if m.sum()>3: C[j]=p[m].mean(0)
        d2=np.sqrt(((p[:,None,:]-C[None])**2).sum(-1)); a_=d2.argmin(1)
        rads=np.array([np.median(d2[a_==j,j]) if (a_==j).sum()>3 else 1e-3 for j in range(k)])
        out.append((f"SphereCluster{k}", SphereCluster(C, np.maximum(rads,1e-3), class_id=cid)))
    return out

def score(prim, ztrue, mask, cam):
    z,ids = render_frame(cam,[prim] if not isinstance(prim,list) else prim)
    pred = ids!=1
    inter=(pred&mask).sum(); union=(pred|mask).sum()
    iou = inter/union if union else 0.0
    both = pred&mask&(ztrue>0)
    if both.sum()<10: return iou, np.nan, np.nan, z, ids
    rel=np.abs(z[both]-ztrue[both])/ztrue[both]
    return iou, float(np.median(rel)), float(np.percentile(rel,90)), z, ids

def ground_plane_params(z,s,f,classes=(0,4)):
    Hh,Ww=192,336; cx,cy=Ww/2-0.5,Hh/2-0.5
    jj,ii=np.mgrid[0:Hh,0:Ww]
    m=np.isin(s,classes)&(z>0)
    A=np.stack([ii[m]-cx,jj[m]-cy,np.ones(m.sum())],1); y=1.0/z[m]; w=np.ones_like(y)
    for _ in range(8):
        pr=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0]
        r=A@pr-y; sd=1.4826*np.median(np.abs(r-np.median(r)))+1e-12
        w=1.0/np.sqrt(1.0+(r/(3*sd))**2)
    a,b,c0=pr; n=np.array([a*f,b*f,c0]); d=1.0/np.linalg.norm(n)
    return n*d, d

def run(n, fr=0, f=206.0, vox=0.30, verbose=True):
    z,s=load(n,fr)
    cam=Camera(focal_px=f, position=np.array([0.,0.,0.]), pitch_deg=0., yaw_deg=0., roll_deg=0.)
    pts=FL.unproject(z,f)
    chosen=[]; report=[]
    for c in sorted(set(np.unique(s).tolist())-{1}):
        m=(s==c)&(z>0)
        if m.sum()<120: continue
        for k,(p,idx) in enumerate(FL.components(pts,m,vox=vox)[:8]):
            smask=np.zeros(m.shape,bool); smask.ravel()[idx]=True
            best=None
            for nm,prim in cands_for(p,c):
                iou,med,p90,_,_=score(prim,z,smask,cam)
                if not np.isfinite(med): med,p90=9.9,9.9
                q=iou-2.0*min(med,1.0)          # silhouette AND depth must both be right
                if best is None or q>best[0]: best=(q,nm,prim,iou,med,p90)
            chosen.append(best[2])
            report.append(dict(cls=int(c),comp=k,npx=int(smask.sum()),
                               bbox=(p.max(0)-p.min(0)).tolist(), z=float(p[:,2].mean()),
                               best=best[1], iou=best[3], med=best[4], p90=best[5]))
            if verbose:
                ext=p.max(0)-p.min(0)
                print(f"  cls{c:2d} {NAME.get(c,'?'):8s} #{k} n={smask.sum():6d} "
                      f"bbox={ext[0]:5.2f}x{ext[1]:5.2f}x{ext[2]:5.2f} z={p[:,2].mean():6.1f}m "
                      f"-> {best[1]:15s} IoU={best[3]:.3f} depth_med={best[4]*100:6.2f}% p90={best[5]*100:7.2f}%")
    return z,s,cam,chosen,report

if __name__=="__main__":
    ex=int(sys.argv[1]) if len(sys.argv)>1 else 28
    ff=float(sys.argv[2]) if len(sys.argv)>2 else 206.0
    print(f"===== example {ex} frame 0, focal {ff}")
    z,s,cam,prims,rep=run(ex,0,ff)
    zr,ir=render_frame(cam,prims)
    both=(z>0)&(zr>0)
    rel=np.abs(zr[both]-z[both])/z[both]
    print(f"\n  FULL SCENE: {len(prims)} primitives")
    print(f"  depth: median rel {np.median(rel)*100:.2f}%  p90 {np.percentile(rel,90)*100:.2f}%  "
          f"covered {both.sum()/ (z>0).sum()*100:.1f}% of their non-sky px")
    print(f"  abs err: median {np.median(np.abs(zr[both]-z[both])):.3f} m  p90 {np.percentile(np.abs(zr[both]-z[both]),90):.3f} m")
    print("  per-class IoU:")
    for c in sorted(set(np.unique(s).tolist())|set(np.unique(ir).tolist())):
        A=(s==c); B=(ir==c); u=(A|B).sum()
        if u==0: continue
        print(f"    class {c:2d} {NAME.get(c,'?'):8s} IoU={(A&B).sum()/u:.3f}  (theirs {A.sum():6d}px, ours {B.sum():6d}px)")
