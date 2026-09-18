import os
import sys, numpy as np, json, os
from multiprocessing import Pool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import focal2 as F
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR)
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
def load(n):
    d=np.load(f"{P}/{n}/condition/000.npz")
    z=d["depth"].astype(np.float64); s=d["semantic_id"]
    if META[n].startswith("uint16"):
        z=np.where(z>0,NEAR*np.exp(L*(1.0-z/65535.0)),0.0); z=np.where(z>=FAR*0.999,0.0,z)
    return z,s
FOCALS=np.arange(90.,611.,10.)
def job(arg):
    n,t0,t1=arg
    try:
        Z,S=load(n)
        zmax=float(np.percentile(Z[t0][Z[t0]>0],95))
        pre=F.icp(Z[t0],S[t0],Z[t1],S[t1],206.,zmax,stride=3)
        if pre is None or not (1.5<pre["ang"]<25.0): return None
        m=F.mask(Z[t0],S[t0],zmax,2); p=F.unproject(Z[t0],206.)[m]
        if len(p)<2000: return None
        c=p.mean(0); u,sv,vt=np.linalg.svd(p-c,full_matrices=False)
        npl=float(np.sqrt((((p-c)@vt[2])**2).mean())/np.sqrt(((p-c)**2).sum(1).mean()))
        if npl<0.05: return None
        tab=F.curve(Z[t0],S[t0],Z[t1],S[t1],FOCALS,zmax=zmax,stride=3)
        v=tab[:,3]
        if np.all(~np.isfinite(v)): return None
        fb,sc=F.argmin_parabolic(tab)
        band=tab[np.nan_to_num(v,nan=9e9)<1.5*sc,0]
        return dict(n=n,t0=t0,t1=t1,f=fb,sharp=float(np.nanmedian(v)/sc),
                    lo=float(band.min()),hi=float(band.max()),rot=float(pre["ang"]),
                    npl=npl,edge=bool(fb<=100 or fb>=600))
    except Exception as e:
        return None
args=[(n,t0,t0+8) for n in range(1,41) for t0 in (0,16,32,48,64,80,96,112) if t0+8<124]
if __name__=="__main__":
    with Pool(12) as pool: res=[r for r in pool.map(job,args) if r]
    json.dump(res,open("focal_all.json","w"),indent=1)
    print(f"{len(res)} usable (example,pair) out of {len(args)} candidates")
    good=[r for r in res if r["sharp"]>2.0 and not r["edge"]]
    print(f"{len(good)} with a sharp minimum (>2x) and not at the grid edge\n")
    byn={}
    for r in good: byn.setdefault(r["n"],[]).append(r)
    for n in sorted(byn):
        fs=np.array([r["f"] for r in byn[n]])
        print(f"  ex{n:3d} n={len(fs):2d}  f = {np.median(fs):6.0f}  [{fs.min():.0f}..{fs.max():.0f}]  "
              f"spread={(fs.max()-fs.min())/np.median(fs)*100:5.0f}%  "
              f"sharp={np.median([r['sharp'] for r in byn[n]]):.1f}x")
    allf=np.array([r["f"] for r in good])
    if len(allf):
        print(f"\nALL sharp estimates: n={len(allf)} median={np.median(allf):.0f} "
              f"p10={np.percentile(allf,10):.0f} p90={np.percentile(allf,90):.0f} "
              f"IQR=[{np.percentile(allf,25):.0f},{np.percentile(allf,75):.0f}]")
