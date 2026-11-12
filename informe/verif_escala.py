# Verificación de la sección 11 del informe (reflexión ×1000, Tablas 12, 13 y 14).
# Calcula el tamaño del conjunto de datos a escala ×1 y ×1000, la probabilidad de que un
# lote no traiga ningún fraude según su tamaño y el error de acumular montos en float32
# frente a float64 (montos simulados con una lognormal de media cercana a 72, semilla 42).
# Uso, desde la raíz del repositorio:  python informe/verif_escala.py
import numpy as np
N=284_807; F=492; C=29; p=F/N
print("tasa",p, "neg/pos",(N-F)/F)
for k in (1,1000):
    n=N*k
    print(f"x{k}: filas {n:,}  float32 {n*C*4/1e9:.3f} GB  float64 {n*C*8/1e9:.3f} GB")
for b in (64,256,512,1024,4096,32768):
    print("batch",b,"esperados",round(b*p,3),"P(0 fraudes)",round((1-p)**b,4))
# float32: suma secuencial de Amount-like
rng=np.random.default_rng(42)
x=rng.lognormal(mean=3.0,sigma=1.6,size=10_000_000).astype(np.float32)  # media ~ 72
for nrep in (1,30):
    xs=np.tile(x,nrep) if nrep>1 else x
    n=xs.size
    s32=np.cumsum(xs,dtype=np.float32)[-1]
    s64=np.sum(xs,dtype=np.float64)
    q32=np.cumsum(xs*xs,dtype=np.float32)[-1]; q64=np.sum(xs.astype(np.float64)**2)
    m32=s32/n; v32=q32/n-m32*m32; m64=s64/n; v64=q64/n-m64*m64
    print(f"n={n:,}: media32={m32:.4f} media64={m64:.4f} err={abs(m32-m64)/m64:.2%} | std32={np.sqrt(max(v32,0)):.3f} std64={np.sqrt(v64):.3f}")
    del xs
print("spacing float32 en 2^24:",np.spacing(np.float32(2**24)),"en 2.5e10:",np.spacing(np.float32(2.5e10)))
