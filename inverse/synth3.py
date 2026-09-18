import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from compiler.render import Camera, Plane, Sphere, Box, Capsule, render_frame
import focal2 as F
rng=np.random.default_rng(0)
prims=[Plane(np.array([0.,0.,0.]),np.array([0.,-1.,0.]),class_id=4)]
for k in range(14):
    c=np.array([rng.uniform(-8,8),-rng.uniform(0.4,2.0),rng.uniform(4,40)])
    prims.append(Box(c,np.array([rng.uniform(.4,1.2),rng.uniform(.4,1.8),rng.uniform(.4,1.2)]),class_id=0))
for k in range(14):
    c=np.array([rng.uniform(-10,10),-rng.uniform(0.5,3.0),rng.uniform(3,45)])
    prims.append(Sphere(c,rng.uniform(0.4,1.5),class_id=5))
focals=np.arange(90.,701.,10.)
import itertools
for ftrue,(dp,dy,dt) in itertools.product((206.,140.,330.,480.),
        [(-4.5,4.0,(0.35,-0.02,0.7))]):
    c0=Camera(focal_px=ftrue,position=np.array([0.,-1.7,0.]),pitch_deg=-8.,yaw_deg=0.)
    c1=Camera(focal_px=ftrue,position=np.array([0.,-1.7,0.])+np.array(dt),pitch_deg=dp,yaw_deg=dy)
    z0,s0=render_frame(c0,prims); z1,s1=render_frame(c1,prims)
    tab=F.curve(z0,s0,z1,s1,focals,zmax=45.0)
    fb,sc=F.argmin_parabolic(tab)
    band=tab[tab[:,3]<1.10*sc,0]
    print(f"true={ftrue:6.1f} -> {fb:7.1f} ratio={fb/ftrue:.3f} score={sc:.5f} band10%=[{band.min():.0f},{band.max():.0f}] rot={np.nanmedian(tab[:,4]):.2f}deg")
    v=tab[:,3]/sc
    print("   f:score/min  "+"  ".join(f"{int(a)}:{b:.2f}" for a,b in zip(tab[::4,0],v[::4])))
