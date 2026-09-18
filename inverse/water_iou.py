"""Self-inconsistency: 3b says water is an exact plane, but the assembled scene
scores water IoU 0.000.  Which of my own results is wrong?
Score the water segment ALONE, then in the assembled scene."""
import sys, numpy as np, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from compiler.render import Camera, Plane, Rect, render_frame
import fitlib as FL, step4
F=206.0
cam=Camera(focal_px=F,position=np.array([0.,0.,0.]),pitch_deg=0.,yaw_deg=0.,roll_deg=0.)
for n in (5,8,39,16,21,18,12):
    z,s=step4.load(n,0)
    m=(s==2)&(z>0)
    if m.sum()<200: print(f"ex{n}: only {m.sum()} water px"); continue
    pts=FL.unproject(z,F); p=pts[m]
    # exact plane through the water points
    c=p.mean(0); u,sv,vt=np.linalg.svd(p-c,full_matrices=False); nrm=vt[2]
    for nm,prim in [("Plane(unbounded)",Plane(c,nrm,class_id=2)),
                    ("PlaneDisc",Plane(c,nrm,class_id=2,radius=float(np.linalg.norm(p-c,axis=1).max()))),
                    ("Rect(horizontal quad)",Rect(y=float(np.median(p[:,1])),x0=float(p[:,0].min()),x1=float(p[:,0].max()),
                                                   z0=float(p[:,2].min()),z1=float(p[:,2].max()),class_id=2))]:
        zr,ir=render_frame(cam,[prim]); pred=ir!=1
        iou=(pred&m).sum()/max((pred|m).sum(),1)
        b=pred&m&(z>0)
        med=np.median(np.abs(zr[b]-z[b])/z[b])*100 if b.sum()>10 else float("nan")
        print(f"ex{n:3d} {m.sum():6d}px  {nm:22s} ALONE IoU={iou:.3f} depth_med={med:6.3f}%")
    # in the assembled scene
    _,_,_,prims,_=step4.run(n,0,F,verbose=False)
    zr,ir=render_frame(cam,prims)
    A=s==2; B=ir==2
    print(f"      -> in assembled scene ({len(prims)} prims): water IoU={(A&B).sum()/max((A|B).sum(),1):.3f}  ours {B.sum()}px theirs {A.sum()}px")
