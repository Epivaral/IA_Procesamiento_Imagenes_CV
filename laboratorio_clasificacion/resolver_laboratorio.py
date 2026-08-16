"""Ejecuta la solución de referencia y guarda el artefacto para inferencia."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
from lib_modelos import (
    ajustar_estandarizador, buscar_hiperparametros, buscar_umbral, cargar_csv,
    division_estratificada, guardar_modelo, metricas, transformar,
)

RAIZ = Path(__file__).resolve().parent

def fila_publica(resultado):
    return {
        "arquitectura": resultado["configuracion"]["arquitectura"],
        "costo_dev": resultado["costo"], "umbral_dev": resultado["umbral"],
        "accuracy_dev": resultado["accuracy"], "precision_dev": resultado["precision"],
        "recall_dev": resultado["recall"], "f1_dev": resultado["f1"],
    }

def main():
    x, y, _ = cargar_csv(RAIZ/"datos_publicos"/"train_1000_desbalanceado.csv")
    indice_train, indice_dev = division_estratificada(y, .20, semilla=31)
    media, desviacion = ajustar_estandarizador(x[indice_train])
    xt = transformar(x[indice_train], media, desviacion)
    xd = transformar(x[indice_dev], media, desviacion)

    configuraciones = [
        {"arquitectura": {"tipo": "mlp", "entrada": 4, "ocultas": [32, 16]},
         "entrenamiento": {"epocas": 100, "batch_size": 64, "learning_rate": .003},
         "costo_fn": 5, "costo_fp": 1},
        {"arquitectura": {"tipo": "mlp_residual_bn", "entrada": 4, "ancho": 32},
         "entrenamiento": {"epocas": 100, "batch_size": 64, "learning_rate": .002},
         "costo_fn": 5, "costo_fp": 1},
    ]
    mejor, resultados = buscar_hiperparametros(configuraciones, xt, y[indice_train], xd, y[indice_dev], semilla=41)
    carpeta = RAIZ/"instructor_privado"/"artefactos"
    guardar_modelo(carpeta/"modelo_mejor.npz", mejor["modelo"], mejor["configuracion"]["arquitectura"], media, desviacion)

    xpriv, ypriv, _ = cargar_csv(RAIZ/"instructor_privado"/"test_200_balanceado.csv")
    ppriv = mejor["modelo"].probabilidad(transformar(xpriv, media, desviacion))
    mpriv = metricas(ypriv, ppriv, mejor["umbral"])
    resumen = {
        "split": {"train": int(len(indice_train)), "dev": int(len(indice_dev)), "clase_1_train": int(y[indice_train].sum()), "clase_1_dev": int(y[indice_dev].sum())},
        "candidatos": [fila_publica(r) for r in resultados],
        "mejor": fila_publica(mejor),
        "evaluacion_privada": mpriv,
    }
    (carpeta/"resumen_solucion.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")

    historia = mejor["historia"]; ejes = range(1, len(historia["loss_train"])+1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ejes, historia["loss_train"], label="train"); ax[0].plot(ejes, historia["loss_dev"], label="dev")
    ax[0].set(xlabel="Época", ylabel="Pérdida BCE ponderada", title="Curvas de pérdida"); ax[0].legend()
    ax[1].plot(ejes, historia["accuracy_dev"], label="Accuracy dev"); ax[1].plot(ejes, historia["f1_dev"], label="F1 dev")
    ax[1].set(xlabel="Época", ylabel="Métrica", title="Desarrollo"); ax[1].legend()
    fig.tight_layout(); fig.savefig(carpeta/"curvas_solucion.png", dpi=160); plt.close(fig)
    print(json.dumps(resumen, indent=2))

if __name__ == "__main__":
    main()
