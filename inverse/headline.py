"""The headline number: single-frame geometric round-trip across examples.
Fit primitives to frame 0, render at their exact settings, score against frame 0.
This is a SELF-FIT ceiling, not a prediction -- stated as such."""
import sys, numpy as np, json
from multiprocessing import Pool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from compiler.render import render_frame
import step4
NAME={0:"terrain",1:"sky",2:"water",4:"road",5:"veg",8:"human",9:"animal",10:"vehicle"}
def job(n):
    try:
        z,s,cam,prims,rep=step4.run(n,0,206.0,verbose=False)
        zr,ir=render_frame(cam,prims)
        b=(z>0)&(zr>0)
        rel=np.abs(zr[b]-z[b])/z[b]; ab=np.abs(zr[b]-z[b])
        io={}
        for c in sorted(set(np.unique(s).tolist())|set(np.unique(ir).tolist())):
            A=s==c; B=ir==c; u=(A|B).sum()
            if u: io[int(c)]=float((A&B).sum()/u)
        return dict(n=n,nprim=len(prims),med=float(np.median(rel)),p90=float(np.percentile(rel,90)),
                    amed=float(np.median(ab)),ap90=float(np.percentile(ab,90)),
                    cov=float(b.sum()/(z>0).sum()),iou=io,
                    dof=sum(getattr(p,'centres',np.zeros((1,3))).size*4//3 if p.__class__.__name__=='SphereCluster' else 9 for p in prims))
    except Exception as e:
        return dict(n=n,err=str(e))
if __name__=="__main__":
    EX=[1,2,3,5,8,15,16,19,21,22,28,29,30,32,35,36,39,40]
    with Pool(9) as p: res=p.map(job,EX)
    print(" ex  prims   depth_med   depth_p90   abs_med   abs_p90   cover   road   veg   human  terrain  water  vehicle")
    for r in res:
        if "err" in r: print(f"{r['n']:3d}  ERROR {r['err'][:60]}"); continue
        g=lambda c: f"{r['iou'].get(c,float('nan')):.2f}"
        print(f"{r['n']:3d}  {r['nprim']:5d} {r['med']*100:9.2f}% {r['p90']*100:10.2f}% "
              f"{r['amed']:8.3f}m {r['ap90']:8.3f}m {r['cov']*100:6.1f}%  {g(4)}  {g(5)}  {g(8)}   {g(0)}    {g(2)}   {g(10)}")
    ok=[r for r in res if "err" not in r]
    med=np.array([r["med"] for r in ok]); p90=np.array([r["p90"] for r in ok])
    print(f"\n N={len(ok)}  depth median-of-medians {np.median(med)*100:.2f}%  "
          f"median p90 {np.median(p90)*100:.1f}%  worst p90 {p90.max()*100:.1f}%")
    for c,nm in NAME.items():
        v=[r["iou"][c] for r in ok if c in r["iou"]]
        if v: print(f"   class {c:2d} {nm:8s} IoU median {np.median(v):.3f} over {len(v)} examples  (min {min(v):.3f} max {max(v):.3f})")
    json.dump(ok,open("headline.json","w"),indent=1)
