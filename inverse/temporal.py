"""The decisive test: did we recover GEOMETRY, or just refit one depth map?

Fit the primitives on frame 0.  Recover the camera motion to frame t from THEIR
own data.  Render the frame-0 scene from the frame-t camera and score it against
their frame t.  Compared against two baselines:
   NULL   : just reuse frame 0's depth map at frame t (a static-camera guess)
   ORACLE : refit the primitives to frame t itself (self-fit ceiling)
A recovered scene must beat NULL and approach ORACLE.  If it only matches NULL,
the 'reconstruction' is a per-frame depth refit wearing a 3D costume.
"""
import sys, numpy as np, json
import os

EXAMPLES = os.environ.get("CWM_EXAMPLES", "examples40")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from dataclasses import dataclass, field
from compiler.render import Camera, render_frame
import focal2 as F2, fitlib as FL, step4

@dataclass
class PoseCam(Camera):
    Rb: np.ndarray = field(default_factory=lambda: np.eye(3))
    def basis(self): return self.Rb

def score(zp, ip, zt, st):
    b=(zp>0)&(zt>0)
    rel=np.abs(zp[b]-zt[b])/zt[b]
    ious={}
    for c in sorted(set(np.unique(st).tolist())|set(np.unique(ip).tolist())):
        A=st==c; B=ip==c; u=(A|B).sum()
        if u: ious[int(c)]=float((A&B).sum()/u)
    return float(np.median(rel)), float(np.percentile(rel,90)), float(b.sum()/(zt>0).sum()), ious

def main(ex=28, f=206.0, ts=(1,2,4,8,16,32,64)):
    print(f"=== ex{ex} focal {f}: fit on frame 0, predict later frames")
    d=np.load(f"{EXAMPLES}/{ex}/condition/000.npz")
    Z=d["depth"].astype(np.float64); S=d["semantic_id"]
    z0,s0,cam0,prims,rep=step4.run(ex,0,f,verbose=False)
    zr0,ir0=render_frame(cam0,prims)
    m,p9,cov,io=score(zr0,ir0,z0,s0)
    print(f"  frame 0 self-fit      : depth med {m*100:6.2f}%  p90 {p9*100:7.2f}%  road IoU {io.get(4,float('nan')):.3f}")
    zmax=float(np.percentile(Z[0][Z[0]>0],95))
    for t in ts:
        if t>=Z.shape[0]: break
        r=F2.icp(Z[0],S[0],Z[t],S[t],f,zmax,stride=2)
        if r is None: print(f"  t={t}: ICP failed"); continue
        R,tv=r["R"],r["t"]          # X_t = R X_0 + tv
        cam=PoseCam(focal_px=f, position=-R.T@tv, Rb=R)
        zr,ir=render_frame(cam,prims)
        m,p9,cov,io=score(zr,ir,Z[t],S[t])
        # NULL baseline: reuse frame 0
        mn,p9n,_,ion=score(z0,s0,Z[t],S[t])
        print(f"  t={t:3d} rot={r['ang']:5.2f}deg trans={r['trans']:5.2f}m | "
              f"OURS med {m*100:6.2f}% p90 {p9*100:7.1f}% roadIoU {io.get(4,float('nan')):.3f} | "
              f"NULL med {mn*100:6.2f}% p90 {p9n*100:7.1f}% roadIoU {ion.get(4,float('nan')):.3f}")
    # ORACLE at the last t
    t=ts[-1]
    if t<Z.shape[0]:
        _,_,camx,primsx,_=step4.run(ex,t,f,verbose=False)
        zr,ir=render_frame(camx,primsx)
        m,p9,cov,io=score(zr,ir,Z[t],S[t])
        print(f"  t={t:3d} ORACLE (refit on that frame): med {m*100:6.2f}% p90 {p9*100:7.1f}% roadIoU {io.get(4,float('nan')):.3f}")
if __name__=="__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 28, float(sys.argv[2]) if len(sys.argv)>2 else 206.0)
