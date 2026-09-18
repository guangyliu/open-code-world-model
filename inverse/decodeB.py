import os
"""Group B (ex 3-27) is uint16_le_fixed_log_gray16_codes with mapping
fixed_log_view_z_0p3m_256m_near_white_gray16_v1.  Test the decode by whether it
makes the ROAD planar (affine 1/z in pixel coords) -- a test it can fail."""
import numpy as np, os
P=os.environ.get("CWM_EXAMPLES","examples40")
H,W=192,336; cx,cy=W/2-0.5,H/2-0.5
jj,ii=np.mgrid[0:H,0:W]
NEAR,FAR=0.3,256.0; L=np.log(FAR/NEAR)
def dec(v, near_white=True, nbits=65535.0):
    t=v/nbits
    if near_white: return NEAR*np.exp(L*(1.0-t))
    return NEAR*np.exp(L*t)
def planar(z,m):
    m=m&(z>0)&np.isfinite(z)
    if m.sum()<500: return None
    A=np.stack([ii[m]-cx,jj[m]-cy,np.ones(m.sum())],1); y=1.0/z[m]; w=np.ones_like(y)
    for _ in range(6):
        p=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0]
        r=A@p-y; s=1.4826*np.median(np.abs(r-np.median(r)))+1e-12
        w=1.0/np.sqrt(1.0+(r/(3*s))**2)
    rel=np.abs(1.0/(A@p)-z[m])/z[m]
    return np.median(rel),np.percentile(rel,90),m.sum(),p
print("ex  variant                 medrel%  p90%    n     zrange(m)")
for n in (8,19,22,5,3,20):
    d=np.load(f"{P}/{n}/condition/000.npz"); v=d["depth"][0].astype(np.float64); s=d["semantic_id"][0]
    for name,z in [("near_white 0.3-256", dec(v,True)),
                   ("near_black 0.3-256", dec(v,False)),
                   ("near_white 0.1-1000", NEAR*0+0.1*np.exp(np.log(1000/0.1)*(1-v/65535.0))),
                   ("linear inv (disp)", np.where(v>0, 65535.0/np.maximum(v,1e-9), np.inf)),
                   ("linear z", v/65535.0*256.0)]:
        r=planar(z,s==4)
        if r is None: continue
        zz=z[(s==4)&np.isfinite(z)]
        print(f"{n:3d} {name:22s} {r[0]*100:7.3f} {r[1]*100:7.2f} {r[2]:6d}  {zz.min():8.2f}..{zz.max():8.2f}")
    print()
