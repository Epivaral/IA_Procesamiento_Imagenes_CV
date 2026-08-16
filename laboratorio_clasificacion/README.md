# Laboratorio: clasificación tabular con MLP

El laboratorio está en español y distingue claramente los materiales públicos de los privados.

- datos_publicos contiene el único conjunto de entrenamiento que se entrega: 1,000 ejemplos, 750 de clase 0 y 250 de clase 1.
- notebooks/laboratorio_estudiante.ipynb es la versión para completar.
- notebooks/laboratorio_resuelto.ipynb es la referencia para docencia.
- notebooks/inferencia_configurable.ipynb reconstruye un modelo desde un diccionario que indica arquitectura y origen de pesos.
- instructor_privado contiene el test final balanceado, los pesos de referencia y la evaluación. No se distribuye.

Ejecute python3 generar_datos.py para generar los CSV con la ecuación analítica, el término de error y una semilla fija. El código sólo necesita Python 3, NumPy y Matplotlib.

La entrega del estudiante debe incluir código, una tabla de búsqueda de hiperparámetros, curvas de entrenamiento y desarrollo, métricas, la justificación del umbral y predicciones sobre el CSV que indique el instructor. El test privado nunca se usa para ajustar hiperparámetros ni umbral.
