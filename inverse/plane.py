import os
import numpy as np, os, json
P=os.environ.get("CWM_EXAMPLES","examples40")
H,W=192,336
jj,ii=np.mgrid[0:H,0:W]

def load(n,shard=0,fr=0):
    cd=f"{P}/{n}/condition"; fs=sorted(os.listdir(cd))
    d=np.load(f"{cd}/{fs[shard]}")
    return d["depth"][fr].astype(np.float64), d["semantic_id"][fr]

def fit_affine_inv(z,mask,iters=6):
    """1/z = a*i + b*j + c  <=> planar surface under ANY pinhole focal. Robust IRLS."""
    m=mask&(z>0)
    A=np.stack([ii[m],jj[m],np.ones(m.sum())],1)
    y=1.0/z[m]
    w=np.ones_like(y)
    for _ in range(iters):
        Aw=A*w[:,None]; p=np.linalg.lstsq(Aw,y*w,rcond=None)[0]
        r=A@p-y; s=1.4826*np.median(np.abs(r-np.median(r)))+1e-12
        w=1.0/np.sqrt(1.0+(r/(3*s))**2)
    zf=1.0/(A@p)
    rel=np.abs(zf-z[m])/z[m]
    inl=np.abs(r)<3*s
    return p,rel,inl,m.sum()

for n,cls,label in [(28,(4,),"road"),(28,(0,4),"grnd+road"),(40,(4,),"road"),(2,(4,),"road"),
                    (36,(4,),"road"),(38,(4,),"road"),(30,(0,),"ground"),(29,(0,),"ground"),
                    (32,(0,),"ground"),(31,(0,),"ground"),(35,(0,),"ground"),(37,(0,),"ground"),
                    (39,(0,),"ground"),(34,(0,),"ground"),(1,(0,),"ground")]:
    z,s=load(n)
    m=np.isin(s,cls)
    if m.sum()<500: print(f"ex{n:2d} {label:10s}: too few px ({m.sum()})"); continue
    p,rel,inl,nn=fit_affine_inv(z,m)
    print(f"ex{n:2d} {label:10s} n={nn:6d} inl={inl.mean():.2f} "
          f"relerr med={np.median(rel)*100:6.2f}% p90={np.percentile(rel,90)*100:7.2f}%  "
          f"a={p[0]:+.3e} b={p[1]:+.3e} c={p[2]:+.3e}")
