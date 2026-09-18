import os
"""The estimator returned 207 and our renderer's default is 206.0.  That is a
coincidence that has to be killed or confirmed before it is believed.

TEST A  gate-focal dependence: the pair-selection step ran ICP at 206.  Re-run
        the WHOLE thing gated at 120 and at 350.  If 207 tracks the gate it is
        an artifact of the initialisation; if it does not move, it is real.
TEST B  injected zoom: a central crop of their depth map, resized back to
        336x192, IS the same scene through a camera of focal k*f.  Feed the
        estimator k=1.5 and k=2 versions of THEIR OWN data.  It must return
        1.5x and 2x whatever it returned at k=1.  This tests the estimator
        end-to-end on real data against a known ground-truth change.
"""
import sys, numpy as np, json
from multiprocessing import Pool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import focal2 as F
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR)
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
H,W=192,336
def load(n):
    d=np.load(f"{P}/{n}/condition/000.npz")
    z=d["depth"].astype(np.float64); s=d["semantic_id"]
    if META[n].startswith("uint16"):
        z=np.where(z>0,NEAR*np.exp(L*(1.0-z/65535.0)),0.0); z=np.where(z>=FAR*0.999,0.0,z)
    return z,s
def zoom(z,s,k):
    """Central crop by k, nearest-resize back: exactly a focal multiplied by k."""
    h2,w2=int(round(H/k)),int(round(W/k))
    y0,x0=(H-h2)//2,(W-w2)//2
    yi=(np.arange(H)*(h2/H)+y0).astype(int).clip(0,H-1)
    xi=(np.arange(W)*(w2/W)+x0).astype(int).clip(0,W-1)
    return z[np.ix_(yi,xi)], s[np.ix_(yi,xi)]
FOCALS=np.arange(90.,901.,10.)
def job(a):
    n,t0,t1,gate,k=a
    try:
        Z,S=load(n)
        z0,s0=Z[t0],S[t0]; z1,s1=Z[t1],S[t1]
        if k!=1.0:
            z0,s0=zoom(z0,s0,k); z1,s1=zoom(z1,s1,k)
        zmax=float(np.percentile(z0[z0>0],95))
        pre=F.icp(z0,s0,z1,s1,gate,zmax,stride=3)
        if pre is None or not (1.5<pre["ang"]<25.0): return None
        tab=F.curve(z0,s0,z1,s1,FOCALS,zmax=zmax,stride=3)
        v=tab[:,3]
        if np.all(~np.isfinite(v)): return None
        fb,sc=F.argmin_parabolic(tab)
        sharp=float(np.nanmedian(v)/sc)
        if sharp<2.0 or fb<=100 or fb>=890: return None
        return (n,gate,k,fb,sharp)
    except Exception: return None
EX=[3,9,15,23,25,26,28,29,39,40,21,35,16,10,22]
TS=[0,32,64,96]
if __name__=="__main__":
    print("TEST A -- does the answer follow the gate focal?")
    args=[(n,t,t+8,g,1.0) for n in EX for t in TS for g in (120.,206.,350.,500.)]
    with Pool(14) as pool: res=[r for r in pool.map(job,args) if r]
    for g in (120.,206.,350.,500.):
        fs=np.array([r[3] for r in res if r[1]==g])
        if len(fs): print(f"  gate f={g:5.0f}: n={len(fs):3d} median={np.median(fs):6.1f} "
                          f"IQR=[{np.percentile(fs,25):.0f},{np.percentile(fs,75):.0f}]")
    print("\nTEST B -- inject a known zoom into THEIR data (truth = k x baseline)")
    args=[(n,t,t+8,206.,k) for n in EX for t in TS for k in (1.0,1.5,2.0)]
    with Pool(14) as pool: res=[r for r in pool.map(job,args) if r]
    base=None
    for k in (1.0,1.5,2.0):
        fs=np.array([r[3] for r in res if r[2]==k])
        if not len(fs): continue
        m=np.median(fs)
        if k==1.0: base=m
        print(f"  zoom k={k:3.1f}: n={len(fs):3d} median f={m:6.1f}  observed ratio={m/base:5.2f}  (expected {k})")
