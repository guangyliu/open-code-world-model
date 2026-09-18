import os
import numpy as np, os
P=os.environ.get("CWM_EXAMPLES","examples40")
H,W=192,336; cx,cy=W/2-0.5,H/2-0.5
jj,ii=np.mgrid[0:H,0:W]
def fit(z,m):
    m=m&(z>0)
    if m.sum()<300: return None
    A=np.stack([ii[m]-cx,jj[m]-cy,np.ones(m.sum())],1); y=1.0/z[m]; w=np.ones_like(y)
    for _ in range(6):
        p=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0]
        r=A@p-y; s=1.4826*np.median(np.abs(r-np.median(r)))+1e-12
        w=1.0/np.sqrt(1.0+(r/(3*s))**2)
    rel=np.abs(1.0/(A@p)-z[m])/z[m]
    return p,np.median(rel),m.sum()
for n,cls in [(28,4),(40,4),(36,4),(2,4)]:
    cd=f"{P}/{n}/condition"; fs=sorted(os.listdir(cd))
    d=np.load(f"{cd}/{fs[0]}"); Z=d["depth"].astype(np.float64); S=d["semantic_id"]
    print(f"=== ex{n} (class {cls})  frame: a  b  c0   -> horizon row j_h, and h/f-family")
    for t in range(0,124,12):
        r=fit(Z[t],S[t]==cls)
        if r is None: print(f"  t={t:3d}  --"); continue
        p,med,nn=r
        a,b,c0=p
        jh=cy-c0/b if abs(b)>1e-12 else float("nan")
        # camera height for a few candidate focals
        hs=[1.0/np.sqrt(f*f*(a*a+b*b)+c0*c0) for f in (150.,206.,300.,450.)]
        print(f"  t={t:3d} n={nn:5d} rel={med*100:5.2f}%  a={a:+.3e} b={b:+.3e} c0={c0:+.4e} "
              f"horizon_row={jh:7.2f}  h(f=150,206,300,450)={hs[0]:.2f},{hs[1]:.2f},{hs[2]:.2f},{hs[3]:.2f}")
