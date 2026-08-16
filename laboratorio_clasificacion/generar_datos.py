"""Genera los datos públicos y privados del laboratorio de forma reproducible."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

SEMILLA = 20260815
COLUMNAS = ("id", "x1", "x2", "x3", "x4", "etiqueta")

def ecuacion_analitica(x, error):
    x1, x2, x3, x4 = x.T
    puntuacion = np.sin(1.25*x1) + 0.72*x2**2 - 0.78*x3 + 0.42*x1*x4 - 0.28*x4**2 + error
    return (puntuacion > 0.82).astype(np.int64)

def muestra_por_clase(rng, negativos, positivos):
    pendientes = {0: negativos, 1: positivos}
    xs, ys = [], []
    while any(pendientes.values()):
        x = rng.uniform(-2.5, 2.5, size=(600, 4))
        y = ecuacion_analitica(x, rng.normal(0.0, 0.48, size=600))
        for clase in (0, 1):
            indice = np.flatnonzero(y == clase)[:pendientes[clase]]
            if len(indice):
                xs.append(x[indice]); ys.append(y[indice]); pendientes[clase] -= len(indice)
    x, y = np.concatenate(xs), np.concatenate(ys)
    orden = rng.permutation(len(y))
    return x[orden], y[orden]

def guardar(ruta, x, y, prefijo):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo); escritor.writerow(COLUMNAS)
        for n, (fila, etiqueta) in enumerate(zip(x, y), 1):
            escritor.writerow([f"{prefijo}_{n:04d}", *[f"{z:.8f}" for z in fila], int(etiqueta)])

def main():
    raiz = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEMILLA)
    x, y = muestra_por_clase(rng, 750, 250)
    guardar(raiz/"datos_publicos"/"train_1000_desbalanceado.csv", x, y, "train")
    ejemplo = raiz/"datos_publicos"/"ejemplo_entrada_inferencia.csv"
    with ejemplo.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(("id", "x1", "x2", "x3", "x4"))
        for n, fila in enumerate(x[:12], 1):
            escritor.writerow([f"ejemplo_{n:04d}", *[f"{z:.8f}" for z in fila]])
    x, y = muestra_por_clase(rng, 100, 100)
    guardar(raiz/"instructor_privado"/"test_200_balanceado.csv", x, y, "test")
    (raiz/"datos_publicos"/"README.md").write_text("# Datos públicos\n\nEl archivo train_1000_desbalanceado.csv contiene 1,000 ejemplos: 750 de la clase 0 y 250 de la clase 1. Las entradas son x1, x2, x3 y x4; etiqueta es el objetivo.\n", encoding="utf-8")
    (raiz/"instructor_privado"/"README.md").write_text("# Material privado del instructor\n\nNo compartir test_200_balanceado.csv. Tiene 200 ejemplos exactamente balanceados, 100 por clase, para la evaluación final.\n", encoding="utf-8")
    print("Datos generados: train=1000 (750/250); test privado=200 (100/100).")

if __name__ == "__main__":
    main()
