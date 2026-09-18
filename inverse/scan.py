import os
import numpy as np, os, json
P=os.environ.get("CWM_EXAMPLES","examples40")
rows=[]
for n in range(1,41):
    cd=f"{P}/{n}/condition"
    fs=sorted(os.listdir(cd))
    d=np.load(f"{cd}/{fs[0]}")
    z=d["depth"]; s=d["semantic_id"]
    tot=s[0].size
    cnt={int(c):int((s[0]==c).sum()) for c in np.unique(s[0])}
    zz=z[z>0]
    rows.append(dict(n=n,shards=len(fs),cls=cnt,
        road=cnt.get(4,0)/tot, ground=cnt.get(0,0)/tot, water=cnt.get(2,0)/tot,
        zmax=float(z.max()), zmed=float(np.median(zz)), z99=float(np.percentile(zz,99))))
for r in rows:
    print("%3d shards=%2d road=%.3f grnd=%.3f water=%.3f zmax=%8.2f zmed=%7.2f z99=%8.2f cls=%s"%(
      r["n"],r["shards"],r["road"],r["ground"],r["water"],r["zmax"],r["zmed"],r["z99"],sorted(r["cls"])))
json.dump(rows,open("scan.json","w"),indent=1)
