import os
"""Two long-open questions, answered with DOF control.

Q1 vegetation: one primitive scaled, a shape library, or arbitrary surface?
   Sweep sphere-cluster lobe count k and watch the error curve.  A real lobed
   crown saturates once k reaches its lobe count; an arbitrary surface keeps
   improving like any function approximator.  CONTROL: run the identical sweep
   on our OWN vegetation_clump render (known answer) and on their human/terrain.

Q2 road & water: separate mesh, or a quad on the ground plane?
   Fit the plane from TERRAIN pixels only, then measure where the ROAD/WATER
   pixels sit relative to it.  A quad on the ground gives ~0 offset and 0 tilt.
"""
import sys, numpy as np, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from compiler.render import Camera, SphereCluster, render_frame, vegetation_clump, Plane
import fitlib as FL
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR)
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
Hh,Ww=192,336; cx,cy=Ww/2-0.5,Hh/2-0.5
jj,ii=np.mgrid[0:Hh,0:Ww]
def load(n,fr=0,sh=0):
    d=np.load(f"{P}/{n}/condition/{sh:03d}.npz")
    z=d["depth"][fr].astype(np.float64); s=d["semantic_id"][fr]
    if META[n].startswith("uint16"):
        z=np.where(z>0,NEAR*np.exp(L*(1.0-z/65535.0)),0.0); z=np.where(z>=FAR*0.999,0.0,z)
    return z,s
def kmeans_cluster(p,k,seed=0,iters=30):
    rng=np.random.default_rng(seed); C=p[rng.choice(len(p),min(k,len(p)),replace=False)].copy()
    for _ in range(iters):
        d=((p[:,None,:]-C[None])**2).sum(-1); a=d.argmin(1)
        for j in range(len(C)):
            m=a==j
            if m.sum()>3: C[j]=p[m].mean(0)
    d=np.sqrt(((p[:,None,:]-C[None])**2).sum(-1)); a=d.argmin(1)
    r=np.array([np.median(d[a==j,j]) if (a==j).sum()>3 else 1e-3 for j in range(len(C))])
    return SphereCluster(C,np.maximum(r,1e-3),class_id=5)
def ksweep(p, ztrue, smask, cam, ks=(1,2,4,8,16,32,64)):
    out=[]
    for k in ks:
        if len(p)<4*k: out.append((k,np.nan,np.nan)); continue
        prim=kmeans_cluster(p,k)
        z,idz=render_frame(cam,[prim]); pred=idz!=1
        u=(pred|smask).sum(); iou=(pred&smask).sum()/u if u else 0
        b=pred&smask&(ztrue>0)
        med=np.median(np.abs(z[b]-ztrue[b])/ztrue[b]) if b.sum()>10 else np.nan
        out.append((k,iou,med))
    return out
def show(tag,res):
    print(f"   {tag:30s} " + " ".join(f"k{k}:{('%.2f'%i) if np.isfinite(i) else '--'}/{('%.1f%%'%(m*100)) if np.isfinite(m) else '--'}" for k,i,m in res))

F=206.0
cam=Camera(focal_px=F,position=np.array([0.,0.,0.]),pitch_deg=0.,yaw_deg=0.,roll_deg=0.)
print("=== Q1  k-sweep, printed as  IoU/median-depth-error  per lobe count k")
# CONTROL: our own vegetation_clump, known to be 18 overlapping spheres
ctrl=vegetation_clump(np.array([0.,-2.0,8.0]),1.2,lobes=18,spread=1.10,seed=3)
zc,sc_=render_frame(cam,ctrl)
mc=(sc_==5)&(zc>0)
pc=FL.unproject(zc,F)[mc]
show("CONTROL our vegetation_clump", ksweep(pc,zc,mc,cam))
for n,cls,tag in [(40,5,"ex40 veg"),(30,5,"ex30 veg"),(35,5,"ex35 veg"),(2,5,"ex2 veg"),
                  (31,5,"ex31 veg"),(28,8,"ex28 human"),(40,8,"ex40 human"),
                  (28,0,"ex28 terrain"),(40,0,"ex40 terrain"),(28,10,"ex28 vehicle")]:
    z,s=load(n); pts=FL.unproject(z,F); m=(s==cls)&(z>0)
    comps=FL.components(pts,m,vox=0.30)
    if not comps: print(f"   {tag}: no component"); continue
    p,idx=comps[0]; sm=np.zeros(m.shape,bool); sm.ravel()[idx]=True
    show(tag+f" (n={len(p)})", ksweep(p,z,sm,cam))

print("\n=== Q2  is road/water a quad on the terrain plane?")
def planefit(z,m):
    m=m&(z>0)
    if m.sum()<400: return None
    A=np.stack([ii[m]-cx,jj[m]-cy,np.ones(m.sum())],1); y=1.0/z[m]; w=np.ones_like(y)
    for _ in range(8):
        pr=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0]
        r=A@pr-y; sd=1.4826*np.median(np.abs(r-np.median(r)))+1e-12
        w=1.0/np.sqrt(1.0+(r/(3*sd))**2)
    rel=np.median(np.abs(1.0/(A@pr)-z[m])/z[m])
    a,b,c0=pr; nv=np.array([a*F,b*F,c0]); d=1.0/np.linalg.norm(nv)
    return nv*d, d, rel, int(m.sum())
for n in (28,40,2,36,38,22,8,19,5,3,20,18,21,12,16,10,39,34):
    z,s=load(n)
    g=planefit(z,s==0); r=planefit(z,s==4); w=planefit(z,s==2)
    if g is None: continue
    line=f"ex{n:3d} terrain: planarity {g[2]*100:5.2f}% h={g[1]:6.2f}m"
    for nm,q in (("road",r),("water",w)):
        if q is None: continue
        ang=np.degrees(np.arccos(np.clip(abs(g[0]@q[0]),-1,1)))
        # how far the other class's points sit off the TERRAIN plane
        m=(s==(4 if nm=="road" else 2))&(z>0)
        X=np.stack([z[m]*(ii[m]-cx)/F, z[m]*(jj[m]-cy)/F, z[m]],-1)
        off=g[1]-X@g[0]
        line+=f" | {nm}: planarity {q[2]*100:5.2f}% tilt_vs_terrain={ang:5.2f}deg off(med)={np.median(off):+6.2f}m iqr={np.percentile(off,75)-np.percentile(off,25):5.2f}m"
    print(line)
