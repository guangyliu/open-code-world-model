import os
import sys, numpy as np, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
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
def run(n,fr=0,f=206.0,vox=0.30,zcap=None):
    z,s=load(n,fr); pts=FL.unproject(z,f)
    if zcap: z=np.where(z<zcap,z,0.0)
    print(f"\n##### ex{n} frame{fr} focal={f} vox={vox}m  fmt={META[n][:8]}")
    for c in sorted(set(np.unique(s).tolist())-{1}):
        m=(s==c)&(z>0)
        if m.sum()<120: continue
        comps=FL.components(pts,m,vox=vox)
        print(f"  class {c} {NAME.get(c,'?'):8s}  {m.sum():6d}px -> {len(comps)} component(s) >=120pts")
        for k,(p,idx) in enumerate(comps[:6]):
            ext=p.max(0)-p.min(0)
            cands=FL.best_fit(p)
            scale=float(np.linalg.norm(ext))
            top="  ".join(f"{nm}:{r:.3f}" for nm,r,_ in cands[:6])
            print(f"     #{k} n={len(p):6d} bbox(WxHxD)={ext[0]:6.2f} x {ext[1]:6.2f} x {ext[2]:6.2f} m  "
                  f"z={p[:,2].mean():6.1f}m diag={scale:6.2f}")
            print(f"        rms residual to primitive (m): {top}")
            print(f"        best={cands[0][0]} rel={cands[0][1]/max(scale,1e-9)*100:.2f}% of diag")
for f in (206.0,):
    run(28,0,f)
    run(40,0,f)
