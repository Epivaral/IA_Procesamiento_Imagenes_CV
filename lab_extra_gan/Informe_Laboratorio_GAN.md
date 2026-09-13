# Informe de laboratorio

## Recuperación del colapso de modos en una GAN

**Estudiante:** Eduardo Pivaral
**Carné:**
**Fecha:** 2026-09-12

---

## 1. Propósito e hipótesis

Todos los experimentos de recuperación parten del mismo punto de control con colapso de modos. La variable independiente es el peso de la pérdida de diversidad λdiv ∈ {0, 0.1, 0.5, 1.0}. La función objetivo del generador es

LG = Ladv + λdiv·Ldiv, con Ldiv = −E[ ‖G(z1) − G(z2)‖1 / (‖z1 − z2‖1 + ε) ]

**1. Antes de observar los resultados, ¿qué valor de λdiv esperaba que funcionara mejor? Explique el beneficio esperado y un posible riesgo.**

Antes de ejecutar el notebook, la expectativa era que λdiv = 0.5 ofreciera el mejor balance. Un valor intermedio penaliza la cercanía entre salidas generadas a partir de puntos latentes distintos sin dominar por completo el objetivo adversarial, lo que en teoría debería ampliar la cobertura de clases sin destruir el realismo de los dígitos. El riesgo esperado era que valores altos (λdiv = 1.0) favorecieran la diversidad de píxeles a costa de la calidad, generando ruido o distorsiones en lugar de dígitos reconocibles, mientras que λdiv = 0 dejaría el colapso esencialmente intacto porque no aporta ninguna presión adicional hacia la diversidad.

---

## 2. Diagnóstico del punto de control colapsado

**2. ¿Qué evidencia visual sugiere colapso de modos? Distinga entre clases ausentes y apariencias repetidas dentro de una clase representada.**

Las muestras del punto de control colapsado (`collapsed_samples.png`) no muestran dígitos reconocibles en ningún caso: las 64 imágenes desplegadas son patrones de manchas claras y oscuras sin la estructura de trazos de un dígito manuscrito. Esto es un colapso más severo que el colapso de modos "entre clases" que el diseño del experimento pretendía inducir (favorecer únicamente los dígitos 1 y 7): aquí no hay clases ausentes en el sentido de que falten algunos dígitos mientras otros se ven bien, sino que la única salida que el clasificador logra reconocer con alta confianza es una textura que asigna casi siempre a la clase "8", sin que esa textura corresponda visualmente a un 8 real. Es decir, la repetición dentro de la "clase representada" es total: todas las imágenes son variaciones mínimas del mismo patrón de ruido estructurado.

**3. Utilice el número de modos y la entropía normalizada para describir el colapso entre clases. ¿Qué información aporta cada medida?**

En el punto de control colapsado, `mode_count = 1/10` y la entropía normalizada es 0.009 (prácticamente 0, frente a un máximo teórico de 1 si las 10 clases estuvieran perfectamente balanceadas). El conteo de modos indica cuántas clases superan el umbral de frecuencia del 1%, es decir, cuántas clases distintas el clasificador llega a detectar entre las muestras generadas; aquí solo una clase (dígito 8) supera ese umbral. La entropía normalizada aporta información adicional sobre qué tan concentrada está la masa de probabilidad entre las clases presentes: un valor cercano a 0 confirma que casi toda la masa recae en una sola clase, mientras que un valor cercano a 1 indicaría un reparto uniforme entre las 10 clases. Ambas medidas coinciden en señalar un colapso extremo, pero la entropía es más sensible a desequilibrios dentro de las clases que sí aparecen, mientras que el conteo de modos solo indica presencia/ausencia binaria por encima del umbral.

Además, el registro impreso del entrenamiento muestra `mass on [1, 7] = 0.000`: la fase de "colapso controlado", diseñada para sesgar al generador hacia los dígitos 1 y 7, no produjo esa distribución. En cambio, el discriminador colapsó (pérdida D = 0.000) desde la época 5 de la fase de preentrenamiento general, antes incluso de iniciar la fase sesgada. Esto indica que el colapso observado en este checkpoint es un colapso de entrenamiento generalizado (el generador venció trivialmente al discriminador) y no un colapso de modo hacia los dígitos objetivo como pretendía el diseño experimental.

**4. ¿Las muestras mostradas permiten identificar confiablemente el colapso dentro de cada clase? Explique qué evidencia adicional se necesita.**

No de forma confiable. La cuadrícula de 64 muestras (`collapsed_samples.png`) permite una inspección visual rápida, pero al no haber dígitos reconocibles resulta imposible evaluar variación de trazo, grosor o estilo dentro de una clase real, porque no existe tal clase reconocible. La medida cuantitativa de apoyo es `within_class_diversity` (distancia promedio entre características del clasificador dentro de cada clase predicha), que en el punto colapsado es 9.53; sin embargo, dado que la clase predicha ("8") no corresponde a un dígito real, esta cifra no refleja variación semántica genuina, sino variación de un patrón de ruido. Se necesitaría evidencia adicional como una muestra mucho más grande, una inspección manual de un subconjunto etiquetado por un humano, o la comparación contra activaciones de imágenes reales de la misma clase para confirmar si la "diversidad" medida corresponde a variación significativa o a artefactos.

---

## 3. Resultados de recuperación

Tabla generada por el notebook (`outputs/gan_mode_collapse/results.csv`):

| Configuration | λdiv | mode_count | normalized_entropy | classifier_confidence | feature_diversity | within_class_diversity |
|---|---|---|---|---|---|---|
| Collapsed start | — | 1 | 0.0094 | 0.9571 | 15.760 | 9.533 |
| lambda=0 | 0.0 | 1 | 0.0202 | 0.9443 | 14.526 | 10.269 |
| lambda=0.1 | 0.1 | 1 | 0.0211 | 0.9422 | 15.803 | 13.344 |
| lambda=0.5 | 0.5 | 1 | 0.0015 | 0.9787 | 16.673 | 16.750 |
| lambda=1.0 | 1.0 | 1 | 0.0168 | 0.9743 | 16.523 | 10.313 |

**5. ¿Qué configuraciones recuperan el mayor número de clases? ¿Un número alto de modos implica necesariamente una distribución equilibrada? Utilice la entropía como evidencia.**

Ninguna configuración recupera más de una clase: `mode_count = 1` en las cinco filas (punto colapsado y las cuatro variantes de λdiv). La figura `class_distributions.png` confirma que, en todos los casos, el clasificador asigna casi el 100% de las muestras a un único dígito (8), con una masa residual mínima en el dígito 3 visible solo para λdiv = 0, 0.1 y 1.0, y prácticamente inexistente para λdiv = 0.5. En principio, un número alto de modos no implica por sí solo una distribución equilibrada, porque el umbral de modo (1%) solo exige presencia mínima por clase y no dice nada sobre qué tan repartida está la masa entre esas clases; la entropía normalizada es la medida que debería usarse para verificar equilibrio. En este experimento la entropía es coherente con el conteo de modos: todos los valores son cercanos a 0 (entre 0.0015 y 0.0211 sobre un máximo de 1), lo que confirma que la recuperación de modos no ocurrió en ninguna configuración. La pérdida de diversidad no logró revertir el colapso severo heredado del checkpoint base.

**6. Compare la confianza del clasificador. Identifique cualquier compromiso aparente entre diversidad y calidad.**

La confianza del clasificador se mantiene alta en todas las configuraciones (0.942–0.979), sin una tendencia clara asociada a λdiv: el valor más alto corresponde a λdiv = 0.5 (0.979) y el más bajo a λdiv = 0.1 (0.942). Esta confianza alta y estable es engañosa como señal de calidad: dado que las imágenes generadas no son dígitos reconocibles (ver Figura 3, `recovery_samples.png`), una confianza alta solo indica que el clasificador asigna consistentemente el mismo patrón de ruido a una clase, no que las muestras sean visualmente correctas. No se observa aquí el compromiso clásico "diversidad vs. calidad" (donde subir λdiv reduciría la confianza al introducir distorsiones), porque en este experimento nunca hubo calidad de partida que sacrificar: el generador nunca salió del régimen de ruido estructurado, con o sin regularización de diversidad.

**7. Compare la diversidad de características con la diversidad dentro de cada clase. ¿Qué configuraciones mejoran ambas medidas y cuáles mejoran principalmente una?**

La diversidad de características (`feature_diversity`) sube ligeramente con λdiv, desde 14.526 en λdiv = 0 hasta un máximo de 16.673 en λdiv = 0.5, y se mantiene similar (16.523) en λdiv = 1.0. La diversidad dentro de clase (`within_class_diversity`) también aumenta con λdiv hasta λdiv = 0.5 (16.750, la más alta de toda la tabla), pero cae de forma notable en λdiv = 1.0 (10.313), muy cerca del valor sin regularización. Solo λdiv = 0.5 mejora ambas medidas de forma consistente y simultánea; λdiv = 0.1 mejora principalmente la diversidad dentro de clase (13.344, por encima del resto salvo 0.5) sin ganancia comparable en diversidad de características global; λdiv = 1.0 no mejora la diversidad dentro de clase pese a tener alta diversidad de características, lo que sugiere que el exceso de peso en la pérdida de diversidad genera variación entre muestras sin necesariamente diversificar dentro del grupo que el clasificador agrupa como misma clase.

**8. Examine las muestras. ¿La mayor diversidad representa variaciones significativas o solamente ruido, distorsiones o dígitos ambiguos? Describa dos ejemplos concretos.**

La mayor diversidad medida numéricamente no se traduce en variaciones significativas. En `recovery_samples.png`, las cinco filas (punto colapsado y las cuatro configuraciones de λdiv) muestran el mismo tipo de patrón: manchas de alto contraste sin la estructura de trazo continuo de un dígito manuscrito. Dos ejemplos concretos:

1. En la fila `lambda=0.5` (la de mayor `feature_diversity` y `within_class_diversity` numérica), las imágenes de las columnas 3 y 4 muestran formas de manchas claramente distintas entre sí en términos de posición de píxeles, pero ninguna de las dos es identificable como un dígito específico; la "diversidad" aquí es diversidad de ruido, no de forma semántica.
2. En la fila `lambda=1.0`, varias imágenes consecutivas (columnas 8 a 11) presentan patrones de manchas con orientación y densidad de píxeles distinta entre sí, lo que eleva la diversidad de características, pero igualmente no corresponden a ningún dígito reconocible; el aumento de variación es indistinguible de ruido aleatorio con distinta textura.

Esto confirma la advertencia del enunciado: un generador puede incrementar la diversidad de píxeles sin producir variación semántica real, y este es exactamente ese caso.

---

## 4. Comportamiento durante el entrenamiento

**9. ¿Por qué las pérdidas totales del generador para distintos valores de λdiv no deben compararse como si midieran exactamente la misma función objetivo?**

La pérdida total del generador es LG = Ladv + λdiv·Ldiv. Cuando λdiv cambia, la función objetivo que el generador minimiza cambia de escala y de composición: para λdiv = 0 la pérdida total es puramente adversarial, mientras que para λdiv = 1.0 hasta la mitad (o más) del valor total puede provenir del término de diversidad. Esto se observa directamente en los históricos: en la época 5, el generador con λdiv = 0 alcanza G_total = 4.080 (igual a su componente adversarial, 4.080), mientras que con λdiv = 1.0 alcanza G_total = 4.470 con un componente adversarial de 4.975 (la resta da la contribución negativa del término de diversidad). Comparar directamente 4.080 contra 4.470 no dice nada sobre cuál generador engaña mejor al discriminador, porque son sumas de cantidades distintas en naturaleza y magnitud; la comparación válida requiere aislar el componente adversarial de cada configuración.

**10. ¿Las curvas de pérdida demuestran por sí solas que se resolvió el colapso de modos? Justifique utilizando los demás resultados.**

No. La pérdida del discriminador se mantiene en 0.000 durante las cinco épocas de recuperación en las cuatro configuraciones (ver `recovery_training.png` y los registros de entrenamiento), lo que indica que el discriminador nunca logró volver a discriminar de forma efectiva entre imágenes reales y generadas en ninguna configuración. Por su parte, la pérdida del generador sube de forma monótona en todos los casos, lo cual por sí solo podría interpretarse como una señal ambigua (¿el generador está mejorando o empeorando?). Solo al cruzar esta información con las métricas de evaluación (mode_count = 1 en todos los casos, entropía cercana a 0, distribuciones de clase dominadas por un único dígito y muestras visualmente no reconocibles como dígitos) se puede concluir que el colapso de modos no se resolvió en ninguna configuración, pese a que las curvas de pérdida por sí solas no lo muestran de forma directa ni permiten esa conclusión sin las métricas de evaluación.

---

## 5. Decisión y limitaciones

**11. Seleccione un valor de λdiv. Defienda la elección con al menos dos resultados cuantitativos y dos observaciones visuales.**

Se selecciona λdiv = 0.5. Resultados cuantitativos: (1) obtiene la mayor diversidad dentro de clase de toda la tabla (16.750, frente a 9.53–13.34 del resto) y la mayor diversidad de características (16.673); (2) obtiene también la mayor confianza del clasificador (0.9787), es decir, no sacrifica estabilidad de clasificación frente a las demás configuraciones. Observaciones visuales: (1) en la distribución de clases (`class_distributions.png`), λdiv = 0.5 es la única configuración en la que la masa residual fuera del dígito dominante es visualmente indistinguible de cero, mostrando el patrón más "limpio" en ese sentido; (2) en la cuadrícula de muestras (`recovery_samples.png`), la fila de λdiv = 0.5 muestra mayor variedad de formas de mancha entre columnas que las filas de λdiv = 0 y λdiv = 0.1, consistente con su mayor diversidad numérica.

Es importante remarcar la limitación central: ninguna configuración, incluida λdiv = 0.5, recuperó modos reales ni produjo dígitos reconocibles (`mode_count = 1/10` en todos los casos). La elección de λdiv = 0.5 se basa en cuál configuración generó la mayor diversidad relativa dentro del régimen de colapso observado, no en que haya resuelto el problema de cobertura de clases planteado por el laboratorio.

**12. Si la fidelidad de las imágenes fuera más importante que la cobertura de clases, ¿cambiaría su elección? Justifique.**

Sí, cambiaría a λdiv = 0.1. Aunque λdiv = 0.5 tiene la mayor confianza del clasificador, esa confianza no es un indicador confiable de fidelidad visual en este experimento, dado que ninguna muestra es un dígito reconocible. Entre las configuraciones que sí muestran algo de estructura visual distinta a puro ruido de alta variación, λdiv = 0.1 introduce la menor perturbación respecto al comportamiento base del generador (λdiv = 0), con una pérdida de diversidad de menor peso relativo en la función objetivo, lo que en teoría preserva mejor cualquier estructura que el generador ya hubiera aprendido antes de sumar el término de diversidad. Sin embargo, dado que ninguna configuración produce imágenes fieles a dígitos reales, esta preferencia es marginal y no constituye una solución al problema de fidelidad visual.

**13. Indique una limitación de utilizar un clasificador de MNIST para medir la cobertura de la GAN y una limitación del diseño experimental.**

Limitación del clasificador: fue entrenado únicamente sobre dígitos reales de MNIST, por lo que ante entradas fuera de esa distribución (como el ruido estructurado producido en este experimento) igualmente emite una predicción de clase con alta confianza (hasta 0.98), sin ningún mecanismo de "no sé" o detección de muestras fuera de distribución; esto puede producir una falsa sensación de que existe una clase claramente identificada cuando en realidad la imagen no corresponde a ningún dígito.

Limitación del diseño experimental: la fase de "colapso controlado" no logró producir el resultado esperado (colapso sesgado hacia los dígitos 1 y 7, con `mass on [1, 7] = 0.000`), porque el discriminador colapsó (pérdida D = 0.000) durante la fase de preentrenamiento general, antes de aplicar el sesgo hacia esos dígitos. Como consecuencia, el checkpoint del que partieron todos los experimentos de recuperación no es representativo del escenario que el laboratorio buscaba estudiar (colapso de modo controlado con clases identificables ausentes), sino de un colapso de entrenamiento adversarial mucho más severo, lo que limita la validez de las conclusiones sobre la eficacia relativa de cada λdiv para ese escenario específico.

**14. Proponga un experimento posterior que fortalezca o cuestione su conclusión. Indique la variable que modificaría y la evidencia que recopilaría.**

Se propone repetir el experimento reduciendo la tasa de aprendizaje del discriminador (o añadiendo un mecanismo de balance, como actualizar el discriminador con menor frecuencia) durante la fase de preentrenamiento general, para evitar que colapse a pérdida 0 antes de iniciar la fase de sesgo hacia los dígitos 1 y 7. La variable a modificar sería `LEARNING_RATE` del discriminador (desacoplándola de la del generador) o el número de pasos de discriminador por paso de generador. La evidencia a recopilar sería la misma tabla de métricas (`mode_count`, `normalized_entropy`, `classifier_confidence`, `feature_diversity`, `within_class_diversity`) aplicada al nuevo checkpoint colapsado y a las cuatro configuraciones de recuperación, verificando en particular que `mass on [COLLAPSE_DIGITS]` sea alta en el punto de partida (evidencia de que el colapso sí ocurrió hacia las clases objetivo) y que la pérdida del discriminador no permanezca en 0 durante la recuperación, lo que indicaría que el discriminador mantiene capacidad de guiar al generador hacia salidas realistas.

---

## Recomendación final

**Valor recomendado:** λdiv = 0.5

**Conclusión en una oración:** Entre las cuatro configuraciones evaluadas, λdiv = 0.5 alcanza la mayor diversidad de características y dentro de clase junto con la mayor confianza del clasificador, pero ninguna configuración recuperó modos reales ni produjo dígitos reconocibles, porque el checkpoint de partida sufrió un colapso de entrenamiento del discriminador (no el colapso de modo hacia dígitos 1/7 que el diseño experimental buscaba inducir), por lo que esta recomendación debe leerse como la mejor opción relativa dentro de un experimento cuyo punto de partida no representa el escenario previsto.
