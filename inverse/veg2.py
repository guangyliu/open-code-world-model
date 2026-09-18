import os
"""v2.  v1's roughness statistic was WRONG: it fitted a quadratic to z, but the
depth of a plane is 1/affine, not affine, so it scored water -- which I had just
proved is an exact plane -- at 9.9% 'roughness'.  The control caught it.
Fix: fit the quadratic to 1/z (exactly affine for a plane), and use 3D connected
components so one blob is one surface rather than several depth layers."""
import sys, numpy as np, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from compiler.render import (Camera, Sphere, Box, Capsule, render_frame, vegetation_clump)
import fitlib as FL
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR); F=206.0
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
def load(n,fr=0):
    d=np.load(f"{P}/{n}/condition/000.npz")
    z=d["depth"][fr].astype(np.float64); s=d["semantic_id"][fr]
    if META[n].startswith("uint16"):
        z=np.where(z>0,NEAR*np.exp(L*(1.0-z/65535.0)),0.0); z=np.where(z>=FAR*0.999,0.0,z)
    return z,s
def descr(z,mask):
    jj,ii=np.nonzero(mask); y=1.0/z[mask]
    A=np.stack([np.ones_like(y),ii,jj,ii*ii,ii*jj,jj*jj],1)
    sol,*_=np.linalg.lstsq(A,y,rcond=None); r=y-A@sol
    # relative depth error of the smooth model == dimensionless roughness
    zf=1.0/np.clip(A@sol,1e-9,None)
    rel=np.abs(zf-z[mask])/z[mask]
    w=ii.max()-ii.min()+1; h=jj.max()-jj.min()+1
    return dict(aspect=float(w/h), rough=float(np.median(rel)), rough90=float(np.percentile(rel,90)),
                px=int(mask.sum()), zmean=float(z[mask].mean()), wpx=int(w), hpx=int(h))
def segs(z,s,cls,vox=0.30,minpx=150):
    pts=FL.unproject(z,F); m=(s==cls)&(z>0)
    out=[]
    for p,idx in FL.components(pts,m,vox=vox,minpts=minpx):
        b=np.zeros(m.shape,bool); b.ravel()[idx]=True; out.append((b,p))
    return out
cam=Camera(focal_px=F,position=np.array([0.,0.,0.]),pitch_deg=0.,yaw_deg=0.,roll_deg=0.)
print("=== CONTROLS (our renderer, shape known) ===")
for nm,prims in [
    ("Box (exactly flat faces)",[Box(np.array([0.,0.,12.]),np.array([1.,1.5,1.]),class_id=5)]),
    ("Sphere r=2 @12m",[Sphere(np.array([0.,0.,12.]),2.0,class_id=5)]),
    ("Ellipsoid 2x3x2 @12m",[Sphere(np.array([0.,0.,12.]),1.0,class_id=5,scale=np.array([2.,3.,2.]))]),
    ("Capsule @12m",[Capsule(np.array([0.,-1.5,12.]),np.array([0.,1.5,12.]),0.8,class_id=5)]),
    ("veg_clump 18 lobes spread1.10",vegetation_clump(np.array([0.,0.,12.]),2.0,lobes=18,spread=1.10,seed=1,class_id=5)),
    ("veg_clump 18 lobes @25m",vegetation_clump(np.array([0.,0.,25.]),3.0,lobes=18,spread=1.10,seed=2,class_id=5)),
    ("veg_clump 8 lobes spread0.75",vegetation_clump(np.array([0.,0.,12.]),2.0,lobes=8,spread=0.75,seed=4,class_id=5)),
]:
    z,s=render_frame(cam,prims)
    ss=segs(z,s,5,minpx=80)
    if not ss: print(f"  {nm}: none"); continue
    d=descr(z,ss[0][0])
    print(f"  {nm:32s} aspect={d['aspect']:5.2f} rough(med)={d['rough']*100:7.3f}% p90={d['rough90']*100:7.3f}% px={d['px']}")
print("\n=== THEIRS, all 40 examples, frame 0, per class ===")
store={}
for cls,nm in ((5,"veg"),(8,"human"),(0,"terrain"),(10,"vehicle"),(2,"water"),(4,"road"),(9,"animal")):
    rr=[]
    for n in range(1,41):
        z,s=load(n)
        for b,p in segs(z,s,cls):
            d=descr(z,b); d["ex"]=n
            ext=p.max(0)-p.min(0); d["bbox"]=[float(x) for x in ext]
            rr.append(d)
    store[nm]=rr
    if not rr: continue
    a=np.array([r["aspect"] for r in rr]); g=np.array([r["rough"] for r in rr])
    print(f"  {nm:8s} N={len(rr):3d} | aspect med {np.median(a):5.2f} p10 {np.percentile(a,10):4.2f} p90 {np.percentile(a,90):5.2f} CV {a.std()/a.mean():.2f}"
          f" | rough med {np.median(g)*100:7.3f}% p10 {np.percentile(g,10)*100:7.3f}% p90 {np.percentile(g,90)*100:7.3f}% CV {g.std()/g.mean():.2f}")
print("\n=== biggest vegetation blobs (3D components) ===")
for r in sorted(store["veg"],key=lambda r:-r["px"])[:18]:
    bb=r["bbox"]
    print(f"  ex{r['ex']:3d} px={r['px']:6d} z={r['zmean']:7.1f}m aspect={r['aspect']:5.2f} "
          f"rough={r['rough']*100:7.3f}% bbox={bb[0]:6.2f}x{bb[1]:6.2f}x{bb[2]:6.2f}m")
json.dump({k:[{kk:vv for kk,vv in r.items()} for r in v] for k,v in store.items()},open("shape_stats.json","w"))
