import os
"""Independent focal cue: a standing human is ~1.7 m tall above the ground plane.

Different mechanism from the rigidity search, so agreement between the two is
evidence and disagreement is a red flag.  Reports, per example, the focal that
makes the tallest class-8 blob exactly 1.70 m -- plus the blob's width/height
aspect, which is what would expose 'this actor is not standing / not a human'.
"""
import numpy as np, json, os
P=os.environ.get("CWM_EXAMPLES","examples40")
H,W=192,336; CX,CY=W/2-0.5,H/2-0.5
jj,ii=np.mgrid[0:H,0:W]
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR)
META={n:json.load(open(f"{P}/{n}/metadata.json"))["depth_format"] for n in range(1,41)}
def load(n,fr=0):
    d=np.load(f"{P}/{n}/condition/000.npz")
    z=d["depth"][fr].astype(np.float64); s=d["semantic_id"][fr]
    if META[n].startswith("uint16"):
        z=np.where(z>0, NEAR*np.exp(L*(1.0-z/65535.0)), 0.0)
        z=np.where(z>=FAR*0.999,0.0,z)
    return z,s
def ground_abc(z,s):
    m=np.isin(s,(0,4))&(z>0)
    if m.sum()<800: return None
    A=np.stack([ii[m]-CX,jj[m]-CY,np.ones(m.sum())],1); y=1.0/z[m]; w=np.ones_like(y)
    for _ in range(8):
        p=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0]
        r=A@p-y; sd=1.4826*np.median(np.abs(r-np.median(r)))+1e-12
        w=1.0/np.sqrt(1.0+(r/(3*sd))**2)
    rel=np.median(np.abs(1.0/(A@p)-z[m])/z[m])
    return p,rel,int(m.sum())
def person_height(z,s,p,f):
    a,b,c0=p
    n=np.array([a*f,b*f,c0]); d=1.0/np.linalg.norm(n); n=n*d   # unit normal, plane n.X = d
    m=(s==8)&(z>0)
    if m.sum()<80: return None
    X=np.stack([z[m]*(ii[m]-CX)/f, z[m]*(jj[m]-CY)/f, z[m]],-1)
    sd=d-X@n                       # signed height above the plane (camera side positive)
    hgt=np.percentile(sd,99.5)-np.percentile(sd,0.5)
    # lateral extent perpendicular to the normal
    t=X-np.outer(X@n,n)
    u,sv,vt=np.linalg.svd(t-t.mean(0),full_matrices=False)
    wid=np.percentile(t@vt[0],99)-np.percentile(t@vt[0],1)
    return hgt,wid,d,int(m.sum())
print("ex  fmt  n_grnd  planerel%   cam_h@f206  person_h@f206  f_for_1.70m   aspect@that_f")
for n in range(1,41):
    try: z,s=load(n)
    except Exception as e: print(n,"err",e); continue
    g=ground_abc(z,s)
    if g is None: print(f"{n:3d}  no ground"); continue
    p,rel,ng=g
    r206=person_height(z,s,p,206.0)
    if r206 is None: print(f"{n:3d}  planerel={rel*100:6.2f}%  no person"); continue
    # solve f such that height==1.70
    fs=np.arange(60.,900.,1.0); hs=[]
    for f in fs:
        r=person_height(z,s,p,f)
        hs.append(np.nan if r is None else r[0])
    hs=np.array(hs)
    k=np.nanargmin(np.abs(hs-1.70)); fbest=fs[k]
    rb=person_height(z,s,p,fbest)
    camh=1.0/np.sqrt(206.0**2*(p[0]**2+p[1]**2)+p[2]**2)
    print(f"{n:3d}  {META[n][:6]}  {ng:6d}  {rel*100:7.2f}   {camh:7.2f}m   {r206[0]:8.2f}m   "
          f"{fbest:7.0f}   w/h={rb[1]/rb[0]:.2f}  (err {abs(hs[k]-1.7):.2f})")
