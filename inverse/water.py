import os
"""'Water planarity 0.00%' needs chasing: exact plane, or a degenerate fit?"""
import numpy as np, json
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR); F=206.0
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
Hh,Ww=192,336; cx,cy=Ww/2-0.5,Hh/2-0.5
jj,ii=np.mgrid[0:Hh,0:Ww]
def load(n,fr=0):
    d=np.load(f"{P}/{n}/condition/000.npz")
    z=d["depth"][fr].astype(np.float64); s=d["semantic_id"][fr]
    raw=d["depth"][fr].astype(np.float64)
    if META[n].startswith("uint16"):
        z=np.where(z>0,NEAR*np.exp(L*(1.0-z/65535.0)),0.0); z=np.where(z>=FAR*0.999,0.0,z)
    return z,s,raw
def pf(z,m):
    m=m&(z>0)
    if m.sum()<300: return None
    A=np.stack([ii[m]-cx,jj[m]-cy,np.ones(m.sum())],1); y=1.0/z[m]; w=np.ones_like(y)
    for _ in range(10):
        pr=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0]
        r=A@pr-y; sd=1.4826*np.median(np.abs(r-np.median(r)))+1e-12
        w=1.0/np.sqrt(1.0+(r/(3*sd))**2)
    rel=np.abs(1.0/(A@pr)-z[m])/z[m]
    a,b,c0=pr; nv=np.array([a*F,b*F,c0]); d=1.0/np.linalg.norm(nv)
    return dict(n=nv*d,d=d,med=np.median(rel),p90=np.percentile(rel,90),p999=np.percentile(rel,99.9),
                mx=rel.max(),cnt=int(m.sum()),abs_med=np.median(np.abs(1.0/(A@pr)-z[m])))
print("ex  class   n      med_rel      p90        p99.9      max      abs_med(m)   depth span")
for n,cls in [(21,2),(18,2),(12,2),(16,2),(8,2),(5,2),(8,4),(5,4),(28,4),(36,4),(19,4),(16,0),(28,0)]:
    z,s,raw=load(n)
    r=pf(z,s==cls)
    if r is None: print(n,cls,"skip"); continue
    zz=z[(s==cls)&(z>0)]
    print(f"{n:3d} {cls:5d} {r['cnt']:7d}  {r['med']:.3e}  {r['p90']:.3e}  {r['p999']:.3e}  {r['mx']:.3e}  {r['abs_med']:.5f}  {zz.min():7.2f}..{zz.max():7.2f}m")
print("\nwater vs road plane angle (both present):")
for n in (8,5,34,18):
    z,s,_=load(n); a=pf(z,s==2); b=pf(z,s==4)
    if a and b:
        ang=np.degrees(np.arccos(np.clip(abs(a["n"]@b["n"]),-1,1)))
        print(f"  ex{n}: water-road tilt {ang:.2f} deg | water d={a['d']:.2f}m road d={b['d']:.2f}m")
print("\ncontrol: same statistic on OUR renderer's exact analytic plane and on a sphere")
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from compiler.render import Camera,Plane,Sphere,render_frame
cam=Camera(focal_px=F,position=np.array([0.,-2.,0.]),pitch_deg=-8.)
zc,sc=render_frame(cam,[Plane(np.array([0.,0.,0.]),np.array([0.,-1.,0.]),class_id=2)])
r=pf(zc,sc==2); print(f"  our exact Plane  : med {r['med']:.3e} p90 {r['p90']:.3e} max {r['mx']:.3e} (n={r['cnt']})")
zc,sc=render_frame(cam,[Sphere(np.array([0.,-2.,12.]),3.0,class_id=2)])
r=pf(zc,sc==2); print(f"  our Sphere r=3m  : med {r['med']:.3e} p90 {r['p90']:.3e} max {r['mx']:.3e} (n={r['cnt']})")
