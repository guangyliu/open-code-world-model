"""The frame-0 scene fails to predict frame t.  Whose fault -- the primitives or
the recovered camera motion?  Warp THEIR OWN frame-0 point cloud by the same
recovered motion and splat it.  That has perfect geometry by construction, so
whatever error it shows is the motion's; anything beyond that is the primitives'."""
import sys, numpy as np, json
import os

EXAMPLES = os.environ.get("CWM_EXAMPLES", "examples40")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import focal2 as F
H,W=192,336; CX,CY=W/2-0.5,H/2-0.5
def splat(X, ids, f):
    """Project a camera-frame cloud to a z-buffer at the released resolution."""
    ok=X[:,2]>1e-3
    X=X[ok]; ids=ids[ok]
    u=np.rint(X[:,0]/X[:,2]*f+CX).astype(int)
    v=np.rint(X[:,1]/X[:,2]*f+CY).astype(int)
    m=(u>=0)&(u<W)&(v>=0)&(v<H)
    u,v,z,i=u[m],v[m],X[m,2],ids[m]
    zb=np.full((H,W),np.inf); ib=np.zeros((H,W),np.uint8)
    order=np.argsort(-z)
    zb[v[order],u[order]]=z[order]; ib[v[order],u[order]]=i[order]
    return np.where(np.isfinite(zb),zb,0.0), np.where(np.isfinite(zb),ib,np.uint8(1))
def run(ex, f=206.0):
    d=np.load(f"{EXAMPLES}/{ex}/condition/000.npz")
    Z=d["depth"].astype(np.float64); S=d["semantic_id"]
    zmax=float(np.percentile(Z[0][Z[0]>0],95))
    pts=F.unproject(Z[0],f); v0=Z[0]>0
    X0=pts[v0]; I0=S[0][v0]
    print(f"ex{ex}: warp their own frame-0 cloud with the recovered motion (perfect geometry)")
    for t in (1,2,4,8,16,32,64):
        if t>=Z.shape[0]: break
        r=F.icp(Z[0],S[0],Z[t],S[t],f,zmax,stride=2)
        if r is None: print(f"  t={t}: ICP failed"); continue
        Xt=X0@r["R"].T + r["t"]
        zw,iw=splat(Xt,I0,f)
        b=(zw>0)&(Z[t]>0)
        rel=np.abs(zw[b]-Z[t][b])/Z[t][b]
        # coverage: warping leaves holes, so also report fill
        print(f"  t={t:3d} rot={r['ang']:5.2f}deg trans={r['trans']:5.2f}m  "
              f"WARP(their geometry) med {np.median(rel)*100:6.2f}% p90 {np.percentile(rel,90)*100:7.1f}% "
              f"fill {b.sum()/(Z[t]>0).sum()*100:5.1f}%")
for ex in (28,40):
    run(ex); print()
