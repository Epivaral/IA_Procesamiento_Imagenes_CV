"""Implementación didáctica con NumPy de MLP, residual MLP+BN y selección."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

CARACTERISTICAS = ("x1", "x2", "x3", "x4")

def cargar_csv(ruta, con_etiqueta=True):
    with Path(ruta).open(encoding="utf-8", newline="") as archivo: filas = list(csv.DictReader(archivo))
    x = np.array([[float(f[c]) for c in CARACTERISTICAS] for f in filas])
    ids = [f["id"] for f in filas]
    return (x, np.array([int(f["etiqueta"]) for f in filas]), ids) if con_etiqueta else (x, ids)

def division_estratificada(y, proporcion_dev=0.20, semilla=17):
    rng = np.random.default_rng(semilla); train=[]; dev=[]
    for c in np.unique(y):
        i = rng.permutation(np.flatnonzero(y == c)); n = round(len(i)*proporcion_dev)
        dev.extend(i[:n]); train.extend(i[n:])
    return rng.permutation(train), rng.permutation(dev)

def ajustar_estandarizador(x):
    media=x.mean(0); desviacion=x.std(0)
    return media, np.where(desviacion < 1e-12, 1.0, desviacion)

def transformar(x, media, desviacion): return (x-media)/desviacion
def sigmoide(z): return 1/(1+np.exp(-np.clip(z, -40, 40)))

def metricas(y, p, umbral=.5):
    pred=(p>=umbral).astype(int); tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
    fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1=2*precision*recall/max(precision+recall,1e-12)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,accuracy=(tp+tn)/len(y),precision=precision,recall=recall,especificidad=tn/max(tn+fp,1),f1=f1)

def buscar_umbral(y,p,costo_fn=5,costo_fp=1):
    filas=[]
    for t in np.linspace(.05,.95,181):
        m=metricas(y,p,t); filas.append(dict(umbral=float(t),costo=costo_fn*m["fn"]+costo_fp*m["fp"],**m))
    return min(filas,key=lambda z:(z["costo"],-z["f1"])),filas

def _bn_adelante(z, entrenando, media, var):
    if entrenando:
        mu=z.mean(0,keepdims=True); va=z.var(0,keepdims=True); media[:]=.9*media+.1*mu; var[:]=.9*var+.1*va
    else: mu,va=media,var
    inv=1/np.sqrt(va+1e-5); hat=(z-mu)*inv
    return hat,(hat,inv)
def _bn_atras(d,c):
    hat,inv=c; n=len(d)
    return inv/n*(n*d-d.sum(0,keepdims=True)-hat*(d*hat).sum(0,keepdims=True))

class MLP:
    arquitectura="mlp"
    def __init__(self, entrada=4, ocultas=(32,16), semilla=0):
        r=np.random.default_rng(semilla); ds=(entrada,)+tuple(ocultas)+(1,)
        self.params={f"{v}{i}": (r.normal(0,np.sqrt(2/ds[i]),(ds[i],ds[i+1])) if v=="W" else np.zeros((1,ds[i+1]))) for i in range(len(ds)-1) for v in ("W","b")}
    def forward(self,x,entrenando=True):
        a=[x]; z=[]; h=x; n=len(self.params)//2-1
        for i in range(n): z.append(h@self.params[f"W{i}"]+self.params[f"b{i}"]); h=np.maximum(z[-1],0); a.append(h)
        return h@self.params[f"W{n}"]+self.params[f"b{n}"],(a,z)
    def probabilidad(self,x): return sigmoide(self.forward(x,False)[0]).ravel()
    def perdida_y_gradiente(self,x,y,peso_positivo=1):
        l,c=self.forward(x); yy=y[:,None]; w=np.where(yy==1,peso_positivo,1); p=sigmoide(l); d=(p-yy)*w/w.sum()
        loss=np.sum(w*(np.maximum(l,0)-l*yy+np.log1p(np.exp(-np.abs(l)))))/w.sum(); a,z=c; n=len(self.params)//2-1; g={f"W{n}":a[-1].T@d,f"b{n}":d.sum(0,keepdims=True)}; dh=d@self.params[f"W{n}"].T
        for i in range(n-1,-1,-1):
            dz=dh*(z[i]>0); g[f"W{i}"]=a[i].T@dz; g[f"b{i}"]=dz.sum(0,keepdims=True); dh=dz@self.params[f"W{i}"].T
        return float(loss),g

class MLPResidualBN:
    arquitectura="mlp_residual_bn"
    def __init__(self,entrada=4,ancho=32,semilla=0):
        r=np.random.default_rng(semilla); ini=lambda a,b:r.normal(0,np.sqrt(2/a),(a,b))
        self.params={"Win":ini(entrada,ancho),"bin":np.zeros((1,ancho)),"W1":ini(ancho,ancho),"b1":np.zeros((1,ancho)),"W2":ini(ancho,ancho),"b2":np.zeros((1,ancho)),"Wout":ini(ancho,1),"bout":np.zeros((1,1))}
        self.media={k:np.zeros((1,ancho)) for k in ("in","1","2")}; self.var={k:np.ones((1,ancho)) for k in ("in","1","2")}
    def forward(self,x,entrenando=True):
        zi=x@self.params["Win"]+self.params["bin"]; ni,ci=_bn_adelante(zi,entrenando,self.media["in"],self.var["in"]); h0=np.maximum(ni,0)
        z1=h0@self.params["W1"]+self.params["b1"]; n1,c1=_bn_adelante(z1,entrenando,self.media["1"],self.var["1"]); a1=np.maximum(n1,0)
        z2=a1@self.params["W2"]+self.params["b2"]; n2,c2=_bn_adelante(z2,entrenando,self.media["2"],self.var["2"]); suma=h0+n2; h=np.maximum(suma,0)
        return h@self.params["Wout"]+self.params["bout"],(x,ni,h0,n1,a1,n2,suma,h,ci,c1,c2)
    def probabilidad(self,x): return sigmoide(self.forward(x,False)[0]).ravel()
    def perdida_y_gradiente(self,x,y,peso_positivo=1):
        l,c=self.forward(x,True); yy=y[:,None]; w=np.where(yy==1,peso_positivo,1); d=(sigmoide(l)-yy)*w/w.sum(); loss=np.sum(w*(np.maximum(l,0)-l*yy+np.log1p(np.exp(-np.abs(l)))))/w.sum()
        x,ni,h0,n1,a1,n2,suma,h,ci,c1,c2=c; g={"Wout":h.T@d,"bout":d.sum(0,keepdims=True)}; ds=(d@self.params["Wout"].T)*(suma>0); dh0=ds.copy()
        dz2=_bn_atras(ds,c2); g["W2"]=a1.T@dz2; g["b2"]=dz2.sum(0,keepdims=True); dn1=(dz2@self.params["W2"].T)*(n1>0)
        dz1=_bn_atras(dn1,c1); g["W1"]=h0.T@dz1; g["b1"]=dz1.sum(0,keepdims=True); dh0+=dz1@self.params["W1"].T
        dzi=_bn_atras(dh0*(ni>0),ci); g["Win"]=x.T@dzi; g["bin"]=dzi.sum(0,keepdims=True)
        return float(loss),g

def crear_modelo(a,semilla=0):
    if a["tipo"]=="mlp": return MLP(a.get("entrada",4),tuple(a.get("ocultas",[32,16])),semilla)
    if a["tipo"]=="mlp_residual_bn": return MLPResidualBN(a.get("entrada",4),a.get("ancho",32),semilla)
    raise ValueError("Arquitectura desconocida")

def entrenar(modelo,x,y,xd,yd,epocas=120,batch_size=64,learning_rate=.002,peso_positivo=None,semilla=0):
    peso_positivo=peso_positivo or float((y==0).sum()/max((y==1).sum(),1)); r=np.random.default_rng(semilla); estado={k:[np.zeros_like(v),np.zeros_like(v)] for k,v in modelo.params.items()}; h={k:[] for k in ("loss_train","loss_dev","accuracy_dev","f1_dev")}; paso=0
    for e in range(epocas):
        for inicio in r.permutation(np.arange(0,len(y),batch_size)):
            ind=np.arange(inicio,min(inicio+batch_size,len(y))); _,g=modelo.perdida_y_gradiente(x[ind],y[ind],peso_positivo); paso+=1
            for k,v in modelo.params.items():
                m,s=estado[k]; m[:]=.9*m+.1*g[k]; s[:]=.999*s+.001*g[k]**2; v[:]-=learning_rate*(m/(1-.9**paso))/(np.sqrt(s/(1-.999**paso))+1e-8)
        lt,_=modelo.perdida_y_gradiente(x,y,peso_positivo); ld,_=modelo.perdida_y_gradiente(xd,yd,peso_positivo); met=metricas(yd,modelo.probabilidad(xd))
        h["loss_train"].append(lt);h["loss_dev"].append(ld);h["accuracy_dev"].append(met["accuracy"]);h["f1_dev"].append(met["f1"])
    return h

def buscar_hiperparametros(configs,x,y,xd,yd,semilla=0):
    filas=[]
    for i,c in enumerate(configs):
        m=crear_modelo(c["arquitectura"],semilla+i); h=entrenar(m,x,y,xd,yd,**c["entrenamiento"],semilla=semilla+i); u,_=buscar_umbral(yd,m.probabilidad(xd),c.get("costo_fn",5),c.get("costo_fp",1)); filas.append(dict(configuracion=c,modelo=m,historia=h,**u))
    return min(filas,key=lambda z:(z["costo"],-z["f1"])),filas

def guardar_modelo(ruta,modelo,arquitectura,media,desviacion):
    ruta=Path(ruta);ruta.parent.mkdir(parents=True,exist_ok=True); d={f"param_{k}":v for k,v in modelo.params.items()};d.update(media=media,desviacion=desviacion,arquitectura_json=np.array(json.dumps(arquitectura)))
    if isinstance(modelo,MLPResidualBN):
        for k in modelo.media:d[f"media_bn_{k}"]=modelo.media[k];d[f"var_bn_{k}"]=modelo.var[k]
    np.savez(ruta,**d)

def cargar_modelo(ruta,arquitectura):
    d=np.load(ruta,allow_pickle=False); m=crear_modelo(arquitectura)
    for k in m.params:m.params[k][:]=d[f"param_{k}"]
    if isinstance(m,MLPResidualBN):
        for k in m.media:m.media[k][:]=d[f"media_bn_{k}"];m.var[k][:]=d[f"var_bn_{k}"]
    return m,d["media"],d["desviacion"]
