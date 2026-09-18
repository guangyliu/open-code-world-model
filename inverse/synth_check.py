"""Validate the focal estimator on renders where f is KNOWN. If it cannot pass
this, nothing it says about the released data is worth anything."""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from compiler.render import Camera, Plane, Sphere, Box, Capsule, render_frame
import focal

rng = np.random.default_rng(0)
def scene():
    p = [Plane(np.array([0.,0.,0.]), np.array([0.,-1.,0.]), class_id=4)]
    for k in range(14):
        c = np.array([rng.uniform(-8,8), -rng.uniform(0.4,2.0), rng.uniform(4,40)])
        p.append(Box(c, np.array([rng.uniform(.4,1.2),rng.uniform(.4,1.8),rng.uniform(.4,1.2)]), class_id=10))
    for k in range(14):
        c = np.array([rng.uniform(-10,10), -rng.uniform(0.5,3.0), rng.uniform(3,45)])
        p.append(Sphere(c, rng.uniform(0.4,1.5), class_id=0))
    for k in range(6):
        a = np.array([rng.uniform(-6,6), -0.1, rng.uniform(3,25)])
        p.append(Capsule(a, a+np.array([0,-1.7,0]), 0.3, class_id=8))
    return p

prims = scene()
focals = np.arange(80., 701., 4.)
for ftrue in (206.0, 140.0, 330.0, 480.0):
    c0 = Camera(focal_px=ftrue, position=np.array([0.,-1.7,0.]), pitch_deg=-8., yaw_deg=0.)
    c1 = Camera(focal_px=ftrue, position=np.array([0.35,-1.72,0.7]), pitch_deg=-4.5, yaw_deg=4.0)
    z0,s0 = render_frame(c0, prims); z1,s1 = render_frame(c1, prims)
    fb, tab, hist = focal.recover(z0,s0,z1,s1,focals, zmax=45.0)
    lo = tab[np.argmin(tab[:,4])]
    print(f"true f={ftrue:6.1f} -> recovered {fb:6.1f}  ratio={fb/ftrue:.3f}  "
          f"npairs={hist[-1][2]} rot={hist[-1][3]:.2f}deg  minscore={lo[4]:.5f}")
    # curve shape
    sc = tab[:,4]; s = sc/sc.min()
    near = tab[s < 1.05, 0]
    print(f"    score within 5% of min over f in [{near.min():.0f},{near.max():.0f}]   hist={[(round(a,1),round(b,1)) for a,b,_,_,_ in hist]}")
