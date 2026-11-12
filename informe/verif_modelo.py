# Verificación de la sección 9 del informe (Tablas 8 y 9).
# Construye la misma red del modelo final (29 -> 64 -> 32 -> 1, BatchNorm, Dropout 0.3),
# cuenta sus parámetros entrenables y mide la memoria de un paso de entrenamiento con un
# lote de 512: parámetros, gradientes, estados de AdamW y tensores que autograd guarda para
# el backward. La función de pérdida solo agrega un tensor del tamaño del lote; la medición
# corresponde casi por completo a las activaciones de la red.
# Uso, desde la raíz del repositorio:  python informe/verif_modelo.py
import torch, torch.nn as nn
torch.manual_seed(42)
ENTRADAS = 29  # V1..V28 + Amount (el equipo descartó Time)
class DetectorFraudeMLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.red=nn.Sequential(nn.Linear(ENTRADAS,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.3),
                            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
    def forward(s,x): return s.red(x)
m=DetectorFraudeMLP()
print("torch",torch.__version__)
for n,p in m.named_parameters(): print(n,tuple(p.shape),p.numel())
P=sum(p.numel() for p in m.parameters() if p.requires_grad); print("entrenables",P)
for n,b in m.named_buffers(): print("buffer",n,b.dtype,b.numel())
opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
x=torch.randn(512,ENTRADAS); y=(torch.rand(512,1)<0.002).float()
saved=[0]
def pack(t): saved[0]+=t.numel()*t.element_size(); return t
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t:t):
    loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([578.]))(m(x),y)
loss.backward(); opt.step()
st=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v) and v.numel()>1)
print("bytes param",P*4,"grad",sum(p.grad.numel()*4 for p in m.parameters()),"adamw estados",st)
print("bytes guardados para backward (batch 512)",saved[0])
