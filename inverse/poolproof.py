import os
"""Direct evidence that the gray16 family is NOT min-positive-depth + copy-semantic:
under that rule a sky pixel is either sky-at-far or replaced wholesale by the near
surface.  Blended values in between can only come from a box MEAN."""
import numpy as np, json
P=os.environ.get("CWM_EXAMPLES","examples40")
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR)
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
print("ex fmt      sky px   sky px with a NON-far depth   their decoded z range")
for n in (3,5,20,23,1,2,28,40,36):
    d=np.load(f"{P}/{n}/condition/000.npz"); v=d["depth"][0]; s=d["semantic_id"][0]
    sky=s==1
    if sky.sum()<50: print(f"{n:3d} {META[n][:7]}  (no sky)"); continue
    if META[n].startswith("uint16"):
        z=np.where(v>0,NEAR*np.exp(L*(1.0-v/65535.0)),FAR)
        bad=sky&(v>0)&(z<FAR*0.5)
        rng=f"{z[sky&(v>0)].min():.1f}..{z[sky&(v>0)].max():.1f} m" if (sky&(v>0)).any() else "-"
    else:
        z=v.astype(float); bad=sky&(z>0); rng=f"{z[sky].min():.1f}..{z[sky].max():.1f} m"
    print(f"{n:3d} {META[n][:7]}  {sky.sum():6d}   {bad.sum():6d} ({bad.sum()/max(sky.sum(),1)*100:5.2f}%)   {rng}")
