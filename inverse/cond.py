import os
"""Which (example, frame pair) are even CONDITIONED for focal recovery?
Need BOTH: static geometry that is not a single plane, AND camera rotation."""
import sys, numpy as np, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import focal2 as F
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR)
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
def load(n,sh=0):
    d=np.load(f"{P}/{n}/condition/{sh:03d}.npz")
    z=d["depth"].astype(np.float64); s=d["semantic_id"]
    if META[n].startswith("uint16"):
        z=np.where(z>0,NEAR*np.exp(L*(1.0-z/65535.0)),0.0); z=np.where(z>=FAR*0.999,0.0,z)
    return z,s
def nonplanarity(z,s,zmax):
    m=F.mask(z,s,zmax,1)
    p=F.unproject(z,206.0)[m]
    if len(p)<500: return 0.0,0,0.0
    c=p.mean(0); u,sv,vt=np.linalg.svd(p-c,full_matrices=False)
    rms=np.sqrt(((p-c)@vt[2]).__pow__(2).mean())
    ext=np.sqrt(((p-c)**2).sum(1).mean())
    return float(rms/ext), int(len(p)), float(ext)
rows=[]
for n in range(1,41):
    Z,S=load(n)
    zz=Z[0][Z[0]>0]
    if zz.size<100: continue
    zmax=float(np.percentile(zz,97))
    best=None
    for t0 in (0,20,40,60,80,100):
        t1=t0+20
        if t1>=Z.shape[0]: break
        np0,npts,ext=nonplanarity(Z[t0],S[t0],zmax)
        r=F.icp(Z[t0],S[t0],Z[t1],S[t1],206.0,zmax,stride=3)
        if r is None: continue
        sc=np0*r["ang"]
        if best is None or sc>best[0]: best=(sc,t0,t1,np0,npts,r["ang"],r["trans"],ext)
    if best: 
        rows.append((n,)+best[1:])
        print(f"ex{n:3d} {META[n][:6]} best pair {best[1]:3d}->{best[2]:3d}  nonplanarity={best[3]:.3f} "
              f"npts={best[4]:6d} rot={best[5]:5.2f}deg trans={best[6]:5.2f}m extent={best[7]:7.1f}m  cond={best[0]:.3f}")
json.dump([[float(x) for x in r] for r in rows],open("cond.json","w"))
