import os
import numpy as np, json, sys
P=os.environ.get("CWM_EXAMPLES","examples40")
d=np.load(f"{P}/1/condition/000.npz")
print(d.files)
z=d["depth"]; s=d["semantic_id"]
print(z.shape,z.dtype,s.shape,s.dtype)
print("depth min/max",z.min(),z.max())
f0=z[0]; s0=s[0]
for c in np.unique(s0):
    m=s0==c
    zz=f0[m]
    print(f"cls {c}: {m.sum():6d} px  z[min,med,max]=({zz.min():.3f},{np.median(zz):.3f},{zz.max():.3f})  zeros={np.sum(zz==0)}")
print("unique depth count frame0:", np.unique(f0).size)
# quantization check
v=np.unique(f0[f0>0])
print("smallest 10 positive depths:", v[:10])
print("largest 5:", v[-5:])
