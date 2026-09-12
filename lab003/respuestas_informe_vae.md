# Informe de laboratorio: selección de la dimensión latente de un autoencoder variacional

## 1. Objetivo del laboratorio

Se desea seleccionar un autoencoder variacional capaz de reconstruir dígitos reconocibles, generar muestras plausibles a partir de la distribución normal estándar, producir interpolaciones graduales y mantener una representación compacta. Se comparan las dimensiones latentes $d_z \in \{2, 8, 16, 64\}$.

## 2. Hipótesis iniciales

### Pregunta 1

¿Qué dimensión latente espera que produzca la menor pérdida de reconstrucción? Explique la relación esperada entre capacidad latente y reconstrucción.

**Respuesta:** Se espera que $d_z = 64$ produzca la menor pérdida de reconstrucción, porque una dimensión latente mayor permite codificar más información sobre trazo, grosor y forma de cada dígito. Al aumentar $d_z$ el cuello de botella es menos restrictivo y el decodificador dispone de más grados de libertad para reproducir detalles finos, de modo que la pérdida de reconstrucción debería disminuir de forma monótona (o casi monótona) al crecer $d_z$, con rendimientos decrecientes conforme la dimensión ya cubre la variabilidad relevante de MNIST.

### Pregunta 2

¿Qué dimensión espera que produzca las muestras más plausibles para $\mathbf{z} \sim \mathcal{N}(0, I)$? Explique por qué la calidad de muestreo puede no seguir el mismo comportamiento que la reconstrucción.

**Respuesta:** Se espera que una dimensión intermedia (por ejemplo $d_z = 8$ o $d_z = 16$) produzca las muestras más plausibles, no necesariamente $d_z = 64$. La reconstrucción usa la media posterior de una imagen real, mientras que el muestreo del prior depende de que todo el espacio latente esté bien cubierto por $q_\phi(\mathbf{z} \mid \mathbf{x})$ y sea coherente con $p(\mathbf{z}) = \mathcal{N}(0, I)$. Con más dimensiones el término de reconstrucción domina la optimización y el modelo puede dejar regiones del espacio latente poco regularizadas o dimensiones con posterior casi igual al prior pero mal organizadas conjuntamente, lo que produce muestras del prior menos coherentes aunque la reconstrucción sea mejor.

### Pregunta 3

Prediga si todas las dimensiones disponibles estarán activas en el modelo con $d_z = 64$. ¿Qué significa que una dimensión se encuentre inactiva?

**Respuesta:** No, se espera que no todas las 64 dimensiones estén activas. Una dimensión inactiva es aquella cuya contribución promedio a la divergencia KL es cercana a cero, es decir, su posterior $q_\phi(z_j \mid \mathbf{x})$ colapsa al prior $\mathcal{N}(0,1)$ independientemente de la entrada $\mathbf{x}$. Esto indica que esa dimensión no transmite información útil sobre la imagen: el decodificador la ignora y el modelo la "apaga" como forma de minimizar el término KL sin costo en reconstrucción.

### Pregunta 4

Escriba una recomendación inicial. Indique qué evidencia podría hacerle cambiar esa recomendación.

**Respuesta:** Recomendación inicial: $d_z = 16$, como punto intermedio entre capacidad suficiente para reconstruir y generar muestras razonables, y un costo computacional y de parámetros moderado. Cambiaría esta recomendación si la tabla cuantitativa mostrara que $d_z = 8$ alcanza una pérdida de reconstrucción casi idéntica a $d_z = 16$ con muchos menos parámetros, o si las muestras del prior e interpolaciones de $d_z = 64$ fueran claramente superiores a pesar de tener dimensiones inactivas.

## 3. Integridad experimental

### Pregunta 5

Registre la configuración y explique por qué deben mantenerse constantes todos los parámetros distintos de la dimensión latente.

**Respuesta:**

| Configuración | Valor |
|---|---|
| Conjunto de datos | MNIST |
| Dimensiones latentes | 2, 8, 16, 64 |
| $\beta$ | 1.0 |
| Épocas | 8 |
| Tamaño de lote | 256 |
| Tasa de aprendizaje | 0.001 |
| Semilla aleatoria | 42 |
| Umbral de actividad | 0.01 |
| Dispositivo utilizado | CPU |

Mantener fijos $\beta$, las épocas, el tamaño de lote, la tasa de aprendizaje, la semilla y el umbral de actividad es necesario para que la única variable que difiere entre los cuatro entrenamientos sea $d_z$. Si se permitiera que otro hiperparámetro variara entre modelos, cualquier diferencia observada en reconstrucción, KL o calidad de muestreo no podría atribuirse de forma inequívoca a la dimensión latente, sino que podría deberse a un entrenamiento más largo, un tamaño de paso distinto o una inicialización diferente, invalidando la comparación controlada.

## 4. Resultados cuantitativos

| Latent dim | Parameters | Test negative ELBO | Test reconstruction | Test KL | Active dimensions | Active fraction | Training seconds |
|---|---|---|---|---|---|---|---|
| 2 | 88,645 | 159.342 | 153.772 | 5.570 | 2 | 1.000 | 284.88 |
| 8 | 145,105 | 115.518 | 99.339 | 16.178 | 8 | 1.000 | 283.95 |
| 16 | 220,385 | 105.613 | 80.834 | 24.779 | 16 | 1.000 | 399.09 |
| 64 | 672,065 | 104.853 | 75.783 | 29.070 | 42 | 0.656 | 824.94 |

### Pregunta 6

Ordene los cuatro modelos según su pérdida de reconstrucción en prueba. ¿Las mejoras entre dimensiones consecutivas son sustanciales o marginales? Incluya los valores medidos.

**Respuesta:** De mayor a menor pérdida de reconstrucción en prueba: $d_z=2$ (153.772) > $d_z=8$ (99.339) > $d_z=16$ (80.834) > $d_z=64$ (75.783). La mejora de $d_z=2$ a $d_z=8$ es sustancial (−54.4 puntos, una reducción de ~35%). La mejora de $d_z=8$ a $d_z=16$ también es notable (−18.5 puntos, ~19%). En cambio, la mejora de $d_z=16$ a $d_z=64$ es marginal (−5.05 puntos, ~6%), a pesar de multiplicar por 4 la dimensión latente y por 3 el número de parámetros. Esto muestra rendimientos claramente decrecientes a partir de $d_z=16$.

### Pregunta 7

¿Cómo cambian la divergencia KL total, el número de dimensiones activas y la fracción activa al aumentar $d_z$? Explique qué revela cada medida sobre la capacidad latente efectiva.

**Respuesta:** La divergencia KL total aumenta con $d_z$ (5.570 → 16.178 → 24.779 → 29.070), pero no de forma proporcional a la dimensión: se multiplica por 32 la dimensión (de 2 a 64) mientras el KL solo se multiplica por ~5.2. El número de dimensiones activas también crece (2 → 8 → 16 → 42), pero la fracción activa se mantiene en 100% hasta $d_z=16$ y cae a 65.6% en $d_z=64$. Esto revela que hasta $d_z=16$ el modelo aprovecha toda la capacidad latente disponible, mientras que en $d_z=64$ una parte sustancial de las dimensiones (22 de 64) queda inactiva porque el modelo ya cuenta con capacidad suficiente para explicar la variabilidad de MNIST; el resto son dimensiones redundantes que el optimizador apaga en lugar de usar.

### Pregunta 8

Compare el número de parámetros y el tiempo de entrenamiento. ¿Aumentar $d_z$ modifica sustancialmente el tamaño total y el costo computacional del modelo?

**Respuesta:** El número de parámetros crece de 88,645 ($d_z=2$) a 672,065 ($d_z=64$), un factor de ~7.6×, impulsado principalmente por las capas totalmente conectadas de codificación/decodificación del vector latente (`mu_head`, `logvar_head`, `decoder_input`), cuyo tamaño escala linealmente con $d_z$. El tiempo de entrenamiento también aumenta de forma notable: de 284.9 s ($d_z=2$) a 824.9 s ($d_z=64$), casi triplicándose, con un salto especialmente marcado entre $d_z=16$ (399.1 s) y $d_z=64$ (824.9 s). Sí, aumentar $d_z$ modifica sustancialmente tanto el tamaño del modelo como el costo computacional, y ese costo no se traduce en una mejora proporcional de la reconstrucción (Pregunta 6).

## 5. Comportamiento durante el entrenamiento

### Pregunta 9

¿Todos los modelos alcanzaron una región razonablemente estable en la última época? Identifique cualquier comparación afectada por convergencia incompleta.

**Respuesta:** Sí, los cuatro modelos muestran curvas de ELBO negativo y de reconstrucción que se aplanan claramente hacia la época 8 (ver `training_curves.png`). $d_z=2$ es el que más lentamente se estabiliza, con una pendiente todavía visible entre las épocas 6 y 8 (162.68 → 159.34 en pérdida de prueba), por lo que su valor final podría subestimar ligeramente su rendimiento potencial si se entrenara por más épocas. Los modelos de $d_z=8$, 16 y 64 muestran curvas prácticamente planas desde la época 5-6, por lo que la comparación entre ellos en la época final es confiable; la comparación que involucra a $d_z=2$ debe interpretarse con la salvedad de que aún no converge por completo.

### Pregunta 10

Describa la relación entre la pérdida de reconstrucción, la divergencia KL y el ELBO negativo. ¿Qué término explica las principales diferencias entre modelos?

**Respuesta:** El ELBO negativo es la suma de la pérdida de reconstrucción y $\beta$ veces la divergencia KL (con $\beta=1$, es la suma directa). Al aumentar $d_z$, la reconstrucción baja (153.77 → 99.34 → 80.83 → 75.78) mientras el KL sube (5.57 → 16.18 → 24.78 → 29.07): son fuerzas opuestas que ambas dependen de la dimensión latente. El término de reconstrucción es el que explica la mayor parte de las diferencias entre modelos: su rango de variación (~78 puntos entre $d_z=2$ y $d_z=64$) es mucho mayor que el del KL (~23.5 puntos), y esto se refleja en que el ELBO negativo total baja fuertemente entre $d_z=2$ y $d_z=16$, impulsado principalmente por la caída en reconstrucción, y luego se aplana entre $d_z=16$ y $d_z=64$ porque las ganancias en reconstrucción ya son pequeñas y se ven parcialmente compensadas por el aumento en KL.

## 6. Análisis de reconstrucción

### Pregunta 11

Identifique al menos dos imágenes en las que aumentar $d_z$ produzca una mejora visual significativa. Describa cambios en trazos, forma, ambigüedad o detalles.

**Respuesta:** En `reconstructions.png`, la columna del dígito "9" (novena columna, etiqueta original ambigua entre 6/9 con trazo curvo) muestra una mejora clara: con $d_z=2$ la reconstrucción es un borrón sin forma definida, con $d_z=8$ y $d_z=16$ ya se distingue un lazo cerrado parecido a un 6/9, y con $d_z=64$ el trazo es más nítido y con mejor definición del gancho inferior. De forma similar, la columna del dígito "4" (quinta columna) pasa de un trazo borroso y grueso en $d_z=2$ a una forma angular reconocible con líneas más delgadas y separadas en $d_z=16$ y $d_z=64$. En ambos casos $d_z=2$ produce una reconstrucción genérica que se parece a un "dígito promedio", mientras que las dimensiones mayores recuperan mejor la identidad específica del trazo original.

### Pregunta 12

¿A partir de qué dimensión las dimensiones adicionales muestran rendimientos decrecientes? Utilice evidencia visual y la pérdida de reconstrucción.

**Respuesta:** Los rendimientos decrecientes aparecen claramente a partir de $d_z=16$. Cuantitativamente, la mejora de reconstrucción de $d_z=16$ a $d_z=64$ es de solo 5.05 puntos (80.83 → 75.78, ~6%), frente a mejoras de 54.4 y 18.5 puntos en los saltos anteriores. Visualmente, en `reconstructions.png` las filas de $d_z=16$ y $d_z=64$ son casi indistinguibles a simple vista para la mayoría de dígitos, mientras que la diferencia entre $d_z=8$ y $d_z=16$ sigue siendo perceptible en algunos trazos. Esto indica que, para esta arquitectura y este número de épocas, $d_z=16$ ya captura la mayor parte de la variabilidad reconstruible de MNIST.

## 7. Análisis de muestras del prior

### Pregunta 13

Compare el realismo y la diversidad de las muestras. ¿Qué modelo produce la mayor proporción de dígitos reconocibles? ¿Cuál muestra mayor variedad visible?

**Respuesta:** En `prior_samples.png`, la fila de $d_z=2$ (primera fila) produce en su mayoría formas borrosas que oscilan entre 0, 2, 3 y 8, con baja variedad de clases y trazos poco definidos, reflejo de que solo 2 dimensiones no alcanzan a modelar las 10 clases con separación clara. Las filas de $d_z=8$ y $d_z=16$ muestran dígitos más nítidos y una mayor proporción reconocible como una clase concreta (0, 1, 4, 7, 8 se distinguen con más claridad). La fila de $d_z=64$ muestra buena nitidez en trazos individuales pero con más ejemplos ambiguos o deformados (formas intermedias entre dos dígitos), consistente con que parte de sus dimensiones activas describen variaciones finas que no siempre corresponden a combinaciones plausibles bajo el prior. En conjunto, $d_z=8$ ó $d_z=16$ ofrecen el mejor balance entre nitidez y reconocibilidad, mientras que $d_z=2$ y $d_z=16$/$d_z=64$ muestran, respectivamente, la menor y mayor diversidad de formas visibles.

### Pregunta 14

¿El modelo con la mejor reconstrucción también produce las mejores muestras del prior? Explique por qué ambos resultados pueden diferir en un VAE.

**Evidencia que debe incluir:** Considere el término de reconstrucción y la necesidad de que los posteriores codificados sean compatibles con el prior utilizado para generar.

**Respuesta:** No necesariamente. $d_z=64$ tiene la mejor reconstrucción (75.78) pero sus muestras del prior en `prior_samples.png` no son claramente superiores a las de $d_z=8$ o $d_z=16$, y presentan más ambigüedad visual. Esto ocurre porque la reconstrucción solo evalúa qué tan bien el decodificador reconstruye $\mathbf{x}$ a partir de la media posterior $\boldsymbol\mu_\phi(\mathbf{x})$ de esa misma imagen, sin exigir que $\mathbf{z}$ provenga literalmente de $\mathcal{N}(0,I)$. El muestreo del prior, en cambio, depende de que la agregación de todos los posteriores $q_\phi(\mathbf{z}\mid\mathbf{x})$ sobre el conjunto de datos se aproxime bien al prior estándar. Con más dimensiones activas, el decodificador puede especializarse en regiones muy específicas del espacio latente que reconstruyen bien las imágenes de entrenamiento pero que no cubren uniformemente el prior, dejando huecos o regiones mal definidas que, al muestrear $\mathbf{z} \sim \mathcal{N}(0,I)$, generan imágenes menos plausibles a pesar de la mejor reconstrucción.

## 8. Análisis de interpolación

### Pregunta 15

¿Qué modelo produce la transición más gradual y plausible? Describa dónde cambia la identidad y si los trazos intermedios permanecen coherentes.

**Respuesta:** En `interpolations.png` (interpolación del dígito 3 al dígito 8), el modelo $d_z=2$ (primera fila) produce una transición muy borrosa donde ambos extremos ya son poco nítidos y el cambio de identidad es difícil de ubicar por la falta de detalle. Los modelos $d_z=8$, $d_z=16$ y $d_z=64$ (filas 2 a 4) muestran una transición más gradual y con trazos más definidos: el "3" se mantiene reconocible hasta aproximadamente $t=0.4$–$0.5$, y a partir de ahí el lazo superior se va cerrando progresivamente hasta formar el "8" hacia $t=0.8$–$1.0$. De estos, $d_z=16$ ofrece la transición visualmente más coherente, sin pasos intermedios ambiguos o deformados, mientras que $d_z=64$ muestra algunos pasos intermedios (alrededor de $t=0.4$–$0.6$) con trazos ligeramente más irregulares.

### Pregunta 16

¿Una interpolación suave demuestra que cada dimensión latente tiene un significado interpretable? Explique la limitación de esa conclusión.

**Respuesta:** No. Una interpolación suave solo demuestra que el decodificador varía de forma continua y sin saltos abruptos en la ruta lineal específica evaluada entre dos puntos $\mathbf{z}_a$ y $\mathbf{z}_b$; esto es una propiedad del mapa aprendido en esa dirección particular del espacio, no una prueba de que cada dimensión individual del vector latente corresponda a un factor semántico identificable (como "inclinación" o "grosor del trazo"). Las dimensiones latentes en un VAE estándar suelen estar entrelazadas (no desentrelazadas), de modo que mover una sola coordenada puede afectar simultáneamente varios atributos visuales a la vez. La suavidad de una interpolación entre dos ejemplos concretos no generaliza a que el espacio completo, o cada eje por separado, tenga una interpretación semántica clara.

## 9. Organización del espacio latente

### Pregunta 17

¿Qué clases forman regiones distinguibles? Identifique clases que se superponen y explique si su similitud visual puede justificarlo.

**Respuesta:** En el panel de $d_z=2$ de `latent_space.png` (el único graficado sin proyección), las clases 0 (azul) y 1 (naranja) forman regiones bastante distinguibles y separadas del resto, ubicadas en extremos opuestos del espacio. La clase 7 (gris) también ocupa una región relativamente propia hacia la derecha. En cambio, las clases 3, 4, 5, 6, 8 y 9 se superponen fuertemente en la zona central. Esta superposición es razonable dado que estos dígitos comparten rasgos de trazo curvo y cerrado (por ejemplo 3/8/9 o 4/9), por lo que en un espacio de solo 2 dimensiones el modelo no tiene capacidad suficiente para separarlos y los agrupa cerca del origen.

### Pregunta 18

¿Por qué la gráfica directa de $d_z = 2$ debe interpretarse de manera diferente a las proyecciones bidimensionales obtenidas mediante PCA?

**Respuesta:** La gráfica de $d_z=2$ muestra las coordenadas reales del espacio latente completo, sin pérdida de información: la distancia y disposición de los puntos corresponde exactamente a $\boldsymbol\mu_\phi(\mathbf{x})$. Las gráficas de $d_z=8$, 16 y 64, en cambio, son proyecciones mediante PCA a 2 componentes principales, que capturan solo la mayor varianza de un espacio de más dimensiones y descartan el resto. Dos puntos que se ven cercanos o superpuestos en la proyección PCA podrían estar bien separados en las dimensiones restantes del espacio latente real; por lo tanto, la aparente falta de separación de clases en los paneles de $d_z=8/16/64$ no implica necesariamente que el espacio latente completo esté peor organizado que el de $d_z=2$, solo que esa organización no es visible en dos dimensiones.

### Pregunta 19

¿Cada modelo utiliza todas las dimensiones disponibles? Compare el número y la proporción de dimensiones activas.

**Respuesta:** No todos. Los modelos con $d_z=2$, $d_z=8$ y $d_z=16$ utilizan el 100% de sus dimensiones (2/2, 8/8 y 16/16 activas, respectivamente), como se observa en `active_dimensions.png`, donde todas las barras superan claramente el umbral de 0.01. El modelo con $d_z=64$ utiliza solo 42 de 64 dimensiones (65.6%), con 22 dimensiones cuya contribución KL promedio es prácticamente nula. Esto muestra que, hasta $d_z=16$, la capacidad latente nominal coincide con la capacidad efectiva, mientras que en $d_z=64$ el modelo tiene capacidad nominal sobrante que no llega a usar.

### Pregunta 20

Si el modelo de 64 dimensiones deja muchas dimensiones inactivas, ¿qué implica para su capacidad nominal y efectiva? ¿Debe considerarse automáticamente un fracaso de entrenamiento?

**Respuesta:** Implica que la capacidad nominal (64 dimensiones disponibles) es mayor que la capacidad efectiva (42 dimensiones realmente usadas): el modelo, dado el objetivo ELBO con $\beta=1$, encontró suficiente con ~42 dimensiones activas para explicar la variabilidad de MNIST bajo esta arquitectura y número de épocas, y las 22 restantes se regularizan hacia el prior porque no reducen más la pérdida de reconstrucción. No debe considerarse automáticamente un fracaso de entrenamiento: este comportamiento es una consecuencia esperada y hasta deseable de la regularización KL en un VAE, que penaliza el uso de dimensiones que no aportan información. Sería un problema solo si el número de dimensiones activas fuera insuficiente para la tarea (colapso posterior severo) o si el modelo no hubiera convergido; ninguno de los dos es el caso aquí, dado que $d_z=64$ obtiene la mejor reconstrucción de los cuatro modelos.

## 11. Recomendación final

### Pregunta 21: dimensión seleccionada

Recomiendo $d_z = 16$ para el escenario planteado.

### Pregunta 22: justificación basada en evidencia

**Respuesta:** Defiendo $d_z=16$ con la siguiente evidencia:

- **Valores cuantitativos:** $d_z=16$ alcanza una pérdida de reconstrucción de 80.83 y un ELBO negativo de 105.61, muy cerca del mejor valor observado ($d_z=64$: 75.78 y 104.85 respectivamente), con una diferencia de solo ~6% en reconstrucción. Usa 220,385 parámetros, un tercio de los 672,065 de $d_z=64$, y entrena en 399.1 s frente a 824.9 s (menos de la mitad del tiempo).
- **Evidencia de reconstrucción:** en `reconstructions.png`, la fila de $d_z=16$ es visualmente casi indistinguible de la de $d_z=64$ para la mayoría de dígitos, incluyendo el caso ambiguo de la columna 9.
- **Realismo y diversidad de muestras del prior:** en `prior_samples.png`, $d_z=16$ produce dígitos nítidos y reconocibles con buena variedad de clases, sin la ambigüedad adicional observada en algunas muestras de $d_z=64$.
- **Comportamiento de interpolaciones:** en `interpolations.png`, $d_z=16$ ofrece la transición más coherente y gradual entre el dígito 3 y el 8, sin pasos intermedios deformados.
- **Dimensiones latentes activas:** $d_z=16$ usa el 100% de sus dimensiones (16/16 activas), es decir, toda su capacidad nominal es capacidad efectiva, a diferencia de $d_z=64$, que deja 22 dimensiones inactivas (65.6% de fracción activa).
- **Compacidad y costo computacional:** con un tercio de los parámetros y menos de la mitad del tiempo de entrenamiento de $d_z=64$, $d_z=16$ logra casi el mismo desempeño, lo que lo hace más eficiente para el escenario de un sistema de exploración de dígitos manuscritos.

### Pregunta 23: debilidad del modelo seleccionado

**Respuesta:** La principal debilidad de $d_z=16$ es que su pérdida de reconstrucción (80.83) sigue siendo mayor que la de $d_z=64$ (75.78), por lo que en aplicaciones donde el detalle fino de reconstrucción sea crítico (por ejemplo, reconstrucción de trazos muy específicos o dígitos atípicos) $d_z=16$ podría perder matices que $d_z=64$ sí captura. La recomendación sigue siendo razonable porque esa diferencia es pequeña (~6%) frente al ahorro sustancial en parámetros (un tercio) y tiempo de entrenamiento (menos de la mitad), y porque $d_z=16$ usa el 100% de su capacidad nominal, mientras que gran parte de la capacidad adicional de $d_z=64$ no se traduce en información latente activa.

### Pregunta 24: escenario alternativo

**Respuesta:** Un escenario realista donde otra dimensión sería preferible es un sistema embebido o de baja latencia donde el costo computacional y el tamaño del modelo son la restricción dominante (por ejemplo, generación de dígitos en un dispositivo con recursos muy limitados o visualización interactiva del espacio latente en 2D para fines educativos). En ese caso seleccionaría $d_z=2$, porque permite graficar el espacio latente directamente sin proyección, tiene el menor número de parámetros (88,645) y el menor tiempo de entrenamiento (284.9 s), a pesar de su reconstrucción notablemente peor; la prioridad ahí es interpretabilidad directa y costo mínimo, no fidelidad de reconstrucción.

## 12. Limitaciones y experimentos posteriores

### Pregunta 25

Identifique al menos tres limitaciones del experimento.

**Respuesta:**

1. **Duración del entrenamiento:** solo 8 épocas por modelo; la curva de $d_z=2$ (Pregunta 9) aún no muestra convergencia completa, por lo que su comparación con los demás modelos podría subestimar su desempeño real.
2. **Variación aleatoria:** cada configuración de $d_z$ se entrenó una sola vez con una única semilla (42); no se reportan repeticiones con semillas distintas, por lo que no se puede distinguir si las diferencias observadas entre modelos (especialmente las marginales, como entre $d_z=16$ y $d_z=64$) son robustas o producto del azar de inicialización y muestreo.
3. **Subjetividad visual:** la evaluación cualitativa de reconstrucciones, muestras del prior e interpolaciones se basa en inspección visual de un número reducido de ejemplos (12-16 imágenes), lo cual introduce subjetividad y puede no representar el comportamiento promedio sobre las 10,000 imágenes de prueba.
4. **Complejidad de MNIST:** MNIST es un conjunto de dígitos simple, en escala de grises y de bajo contraste estructural; las conclusiones sobre la dimensión latente óptima no necesariamente se generalizan a conjuntos de datos más complejos (imágenes a color, texturas, escenas naturales).
5. **Arquitectura y umbral de actividad fijos:** la arquitectura convolucional y el umbral de actividad (0.01) se mantuvieron fijos arbitrariamente; un umbral distinto cambiaría el conteo de dimensiones activas, y una arquitectura con mayor capacidad en el decodificador podría alterar el punto en que aparecen los rendimientos decrecientes.

### Pregunta 26

Proponga un experimento posterior que fortalezca la recomendación. Indique la variable que cambiaría, las cantidades que mediría y qué resultado apoyaría o contradiría su decisión.

**Respuesta:** Propongo repetir el entrenamiento de los cuatro modelos ($d_z \in \{2,8,16,64\}$) con al menos 3 semillas aleatorias distintas y un número mayor de épocas (por ejemplo 20, hasta confirmar convergencia en las curvas de pérdida de prueba). La variable que cambiaría es la semilla aleatoria (manteniendo el resto de la configuración fija) y el número de épocas. Mediría la media y desviación estándar de la pérdida de reconstrucción, el ELBO negativo y la fracción de dimensiones activas en test, para cada $d_z$. Si la diferencia de reconstrucción entre $d_z=16$ y $d_z=64$ (actualmente ~6%) resultara menor que la desviación estándar entre semillas, se reforzaría la recomendación de $d_z=16$ por ser estadísticamente equivalente a $d_z=64$ con menor costo. Si, en cambio, $d_z=64$ mostrara una ventaja consistente y superior a la variación entre semillas, junto con una fracción activa estable por encima del 65.6%, eso contradiría la recomendación actual y favorecería $d_z=64$.

## 13. Conclusión

Con base en el experimento controlado sobre MNIST, se recomienda $d_z=16$ como la dimensión latente para el autoencoder variacional del escenario planteado. Cuantitativamente, este modelo alcanza una pérdida de reconstrucción de 80.83 y un ELBO negativo de 105.61 en el conjunto de prueba, valores muy cercanos al mejor resultado observado ($d_z=64$: 75.78 y 104.85, respectivamente), con una diferencia de solo alrededor del 6%, pero usando un tercio de los parámetros (220,385 frente a 672,065) y menos de la mitad del tiempo de entrenamiento (399.1 s frente a 824.9 s). Cualitativamente, $d_z=16$ produce reconstrucciones casi indistinguibles de las de $d_z=64$, muestras del prior nítidas y reconocibles, e interpolaciones graduales y coherentes entre dígitos distintos. Además, utiliza el 100% de sus dimensiones latentes (16 de 16 activas), a diferencia de $d_z=64$, que deja 22 de 64 dimensiones inactivas, evidenciando capacidad nominal desperdiciada. El principal compromiso observado es que $d_z=16$ sacrifica una pequeña mejora de reconstrucción disponible en $d_z=64$ a cambio de una reducción sustancial en tamaño del modelo y costo computacional, un intercambio razonable cuando ninguna métrica individual domina la decisión y se prioriza eficiencia junto con calidad generativa e interpretativa del espacio latente.
