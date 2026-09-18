import os
"""Is their vegetation one primitive scaled, a shape library, or arbitrary mesh?

Uses descriptors that survive the focal uncertainty:
  aspect   = image-plane width/height of the blob          (f-invariant)
  rough    = RMS of depth after removing a QUADRATIC surface, / mean depth
             i.e. how bumpy the visible surface is, dimensionless  (f-invariant)
A single smooth primitive scaled to any size gives ONE (aspect, rough) point up
to viewing angle.  A shape library gives a few clusters.  An arbitrary mesh
gives a continuum with high roughness.
CONTROLS are rendered by our own renderer, so the reference values are known.
"""
import sys, numpy as np, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from scipy import ndimage
from compiler.render import (Camera, Sphere, Box, Capsule, Plane, render_frame,
                             vegetation_clump, SphereCluster)
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR); F=206.0
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
Hh,Ww=192,336
def load(n,fr=0):
    d=np.load(f"{P}/{n}/condition/000.npz")
    z=d["depth"][fr].astype(np.float64); s=d["semantic_id"][fr]
    if META[n].startswith("uint16"):
        z=np.where(z>0,NEAR*np.exp(L*(1.0-z/65535.0)),0.0); z=np.where(z>=FAR*0.999,0.0,z)
    return z,s
def blobs(z,s,cls,minpx=150):
    m=(s==cls)&(z>0)
    lab,nl=ndimage.label(m)
    out=[]
    for k in range(1,nl+1):
        b=lab==k
        if b.sum()<minpx: continue
        # split off depth-discontinuous parts crudely: keep the modal depth shell
        zz=z[b]; lo,hi=np.percentile(zz,[2,98])
        b=b&(z>=lo*0.7)&(z<=hi*1.4)
        if b.sum()<minpx: continue
        out.append(b)
    return out
def descr(z,b):
    jj,ii=np.nonzero(b)
    w=ii.max()-ii.min()+1; h=jj.max()-jj.min()+1
    zz=z[b]
    A=np.stack([ii*0+1.,ii,jj,ii*ii,ii*jj,jj*jj],1)
    sol,*_=np.linalg.lstsq(A,zz,rcond=None)
    r=zz-A@sol
    return dict(aspect=w/h, rough=float(np.sqrt((r**2).mean())/zz.mean()),
                px=int(b.sum()), zmean=float(zz.mean()), wpx=int(w), hpx=int(h),
                zspan=float(zz.max()-zz.min())/float(zz.mean()))
print("=== CONTROLS (our renderer, known shapes) ===")
cam=Camera(focal_px=F,position=np.array([0.,0.,0.]),pitch_deg=0.,yaw_deg=0.,roll_deg=0.)
ctrls={}
for nm,prims in [
    ("Sphere r2 @12m",[Sphere(np.array([0.,0.,12.]),2.0,class_id=5)]),
    ("Ellipsoid @12m",[Sphere(np.array([0.,0.,12.]),1.0,class_id=5,scale=np.array([2.,3.,2.]))]),
    ("Box 2x3x2 @12m",[Box(np.array([0.,0.,12.]),np.array([1.,1.5,1.]),class_id=5)]),
    ("Capsule @12m",[Capsule(np.array([0.,-1.5,12.]),np.array([0.,1.5,12.]),0.8,class_id=5)]),
    ("vegetation_clump 18 lobes",vegetation_clump(np.array([0.,0.,12.]),2.0,lobes=18,spread=1.10,seed=1,class_id=5)),
    ("vegetation_clump 18 @25m",vegetation_clump(np.array([0.,0.,25.]),3.0,lobes=18,spread=1.10,seed=2,class_id=5)),
    ("vegetation_clump 8 lobes",vegetation_clump(np.array([0.,0.,12.]),2.0,lobes=8,spread=0.75,seed=4,class_id=5)),
]:
    z,s=render_frame(cam,prims)
    for b in blobs(z,s,5,80):
        d=descr(z,b); ctrls[nm]=d
        print(f"  {nm:28s} aspect={d['aspect']:5.2f} rough={d['rough']*100:6.3f}% zspan={d['zspan']*100:6.2f}% px={d['px']}")
        break
print("\n=== THEIRS: every vegetation blob >=150px in frame 0 of all 40 ===")
rows=[]
for n in range(1,41):
    z,s=load(n)
    for b in blobs(z,s,5):
        d=descr(z,b); d["ex"]=n; rows.append(d)
rows.sort(key=lambda r:-r["px"])
for r in rows[:28]:
    print(f"  ex{r['ex']:3d} px={r['px']:6d} {r['wpx']:3d}x{r['hpx']:3d}px z={r['zmean']:7.1f}m "
          f"aspect={r['aspect']:5.2f} rough={r['rough']*100:7.3f}% zspan={r['zspan']*100:7.1f}%")
a=np.array([r["aspect"] for r in rows]); rg=np.array([r["rough"] for r in rows])
print(f"\n  N={len(rows)} vegetation blobs")
print(f"  aspect : median {np.median(a):.2f}  p10 {np.percentile(a,10):.2f}  p90 {np.percentile(a,90):.2f}  CV {a.std()/a.mean():.2f}")
print(f"  rough  : median {np.median(rg)*100:.3f}%  p10 {np.percentile(rg,10)*100:.3f}%  p90 {np.percentile(rg,90)*100:.3f}%  CV {rg.std()/rg.mean():.2f}")
print("\n=== same statistic on OTHER classes, for scale ===")
for cls,nm in ((8,"human"),(0,"terrain"),(10,"vehicle"),(2,"water"),(4,"road")):
    rr=[]
    for n in range(1,41):
        z,s=load(n)
        for b in blobs(z,s,cls): rr.append(descr(z,b))
    if not rr: continue
    a=np.array([r["aspect"] for r in rr]); g=np.array([r["rough"] for r in rr])
    print(f"  {nm:8s} N={len(rr):3d} aspect med {np.median(a):5.2f} CV {a.std()/a.mean():.2f} | "
          f"rough med {np.median(g)*100:7.3f}% p10 {np.percentile(g,10)*100:7.3f}% p90 {np.percentile(g,90)*100:7.3f}% CV {g.std()/g.mean():.2f}")
json.dump(rows,open("veg_rows.json","w"))
