import os
import sys, numpy as np, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import focal2 as F
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR)
import json
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
def load(n,shard=0):
    d=np.load(f"{P}/{n}/condition/{shard:03d}.npz")
    z=d["depth"].astype(np.float64); s=d["semantic_id"]
    if META[n].startswith("uint16"):
        z=np.where(z>0, NEAR*np.exp(L*(1.0-z/65535.0)), 0.0)
        z=np.where(z>=FAR*0.999, 0.0, z)
    return z,s
focals=np.arange(100.,621.,10.)
for n in (28,40,36,2,30,32):
    Z,S=load(n)
    zz=Z[0][Z[0]>0]; zmax=float(np.percentile(zz,85))
    print(f"=== ex{n}  zmax={zmax:.1f}m  fmt={META[n][:12]}")
    for t0,t1 in ((0,12),(24,36),(60,72),(96,108)):
        tab=F.curve(Z[t0],S[t0],Z[t1],S[t1],focals,zmax=zmax,stride=3)
        if np.all(np.isnan(tab[:,3])): print(f"  {t0}->{t1}: no data"); continue
        fb,sc=F.argmin_parabolic(tab)
        band=tab[np.nan_to_num(tab[:,3],nan=9e9)<1.5*sc,0]
        print(f"  {t0:3d}->{t1:3d}  f={fb:6.1f}  score={sc:.5f}  band1.5x=[{band.min():.0f},{band.max():.0f}]  "
              f"rot={np.nanmedian(tab[:,4]):.2f}deg  sharp={np.nanmedian(tab[:,3])/sc:.1f}x")
