import os
import numpy as np, os
P=os.environ.get("CWM_EXAMPLES","examples40")
def look(n):
    cd=f"{P}/{n}/condition"; fs=sorted(os.listdir(cd))
    d=np.load(f"{cd}/{fs[0]}"); z=d["depth"][0]; s=d["semantic_id"][0]
    integral = np.all(z==np.round(z))
    v=np.unique(z)
    print(f"--- ex{n}: integral={integral} nuniq={v.size} min={v.min()} max={v.max()}")
    print("   first8:",v[:8]); print("   last8:",v[-8:])
    # gaps between consecutive unique values
    dv=np.diff(v)
    print("   diff min/med/max: %.4g %.4g %.4g"%(dv.min(),np.median(dv),dv.max()))
    for c in np.unique(s):
        zz=z[s==c]; print(f"   cls{c}: n={zz.size} range {zz.min():.3f}..{zz.max():.3f}")
for n in (5,9,15,21,3,1,28):
    look(n)
