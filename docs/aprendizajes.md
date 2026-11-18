# Lo que aprendimos evaluando Banhammer

Banhammer lee una carta de Magic: The Gathering y dice si es `ok`, `watch` (vigilar) o `likely_ban` (probablemente baneable). Para eso busca cartas y reglas relacionadas en un índice vectorial (RAG) y le pide a un LLM, el "juez", un reporte que cite lo que encontró.

Este documento junta lo que fuimos aprendiendo en las sucesivas evaluaciones: qué medíamos mal, qué descubrimos del juez, qué descubrimos de la búsqueda y qué cambiamos en consecuencia. Las primeras conclusiones sobre el RAG (puntos 5 a 8) las corrigió la medición sin memoria de la sección 6. Los detalles están en [la evaluación con el juez calibrado](resultados-eval-etapa-2-juez-v2.md), [el análisis de recall](resultados-analisis-recall.md), [el plan de evaluación](plan-evaluacion.md) y [la comparación contra la línea base sin RAG](resultados-eval-v4.md).

Todo se hizo con `gpt-4o-mini` y `text-embedding-3-small`, y con un presupuesto total de unos pocos dólares de API.

## Resumen

1. **La primera métrica medía el criterio del gold set, no el sistema.** La exactitud de 3 clases dependía de dónde se pone el corte de `watch`, que no es un dato verificable. La cambiamos por una métrica binaria (baneada o no) donde responder siempre lo mismo da 0,5.
2. **No todos los bans se pueden deducir del texto.** Muchas cartas se banean por cómo dominaron los torneos (metagame), no por lo que dicen. Las separamos de la métrica principal.
3. **El juez es muy ruidoso.** Con tres corridas idénticas, 14 de 37 cartas cambiaron de veredicto. Toda comparación de una sola corrida necesita una regla de decisión fijada de antemano.
4. **Calibrar el prompt del juez rindió más que cualquier mejora de la búsqueda**, y costó prácticamente nada.
5. **El RAG mejora poco el veredicto, y el RAG más simple fue el mejor.** El juez sin búsqueda queda al nivel del azar (0,52). Con una búsqueda simple, la primera corrida dio 0,76, pero al repetirla dio 0,57 y 0,60: el promedio es 0,64, y la mejora (0,12) queda por debajo del umbral. Las búsquedas elaboradas dieron entre 0,48 y 0,60.
6. **El contexto del juez estaba mal armado.** Una sola lista obligaba a elegir entre cartas comparables y candidatas a combo, y le llegaban unas 60 reglas, casi todas irrelevantes. Esto llevó a los dos últimos cambios.
7. **Arreglar el contexto no alcanzó.** Los dos cambios empataron o empeoraron. La búsqueda casi nunca encuentra la pieza del combo (recall@10 cerca de 0).
8. **Pero aunque la búsqueda fuera perfecta, el juez no mejoraría.** Le dimos al juez las piezas correctas del gold set: las nombró en el 72 % de los casos, pero el acierto fue 0,58, igual que sin ellas. Ve el combo, pero no sabe distinguir un combo que merece ban de uno legal. **El límite hoy es el juez, no la búsqueda.**
9. **La línea base sin RAG era injusta, y el gold set está contaminado por memoria.** Sin la exigencia de citas, el LLM solo saca 0,69, al nivel de `naive`. Pero el modelo reconoce 16 de las 41 cartas por su texto anonimizado: estaban en sus datos de entrenamiento.
10. **Con cartas inventadas, el RAG simple es lo mejor: 0,89 contra 0,72 del LLM solo.** En 9 pares de cartas ficticias (una versión rota y otra justa), la búsqueda trae como precedente la carta baneada en la que se inspira la versión rota. En el gold set eso no se veía, porque la carta evaluada es la propia baneada y se excluye de los resultados. **El valor del RAG está en los precedentes baneados, no en las piezas de combo.**
11. **Decidir con una regla fija no arregla al juez.** Si el juez estima rasgos concretos (turno en que gana, piezas, poder por costo) y una regla decide, nunca banea una carta legal pero casi no detecta nada: `gpt-4o-mini` estima mal los rasgos. El problema es razonar la mecánica, no dónde poner el corte.
12. **Un juez más capaz va en la dirección correcta, pero los pares no tienen margen para medirlo.** Con el mismo RAG y el mismo prompt, `gpt-5.4-mini` como juez acertó los 9 pares en la primera corrida (1,00) y 7 en la segunda (0,83). El promedio, 0,92, le saca 0,03 a `naive` con `gpt-4o-mini`. Pero `naive` ya saca 0,89, así que ningún juez podría superar el umbral de 0,15 en este set. Separa bien el par de Dark Depths; el de Thopter (loop infinito contra incremental) sigue siendo una moneda al aire. **Otra vez, la primera corrida exageró.** Con `oracle_pieces`, que sí tiene margen, el juez caro saca 0,62, lo mismo que `gpt-4o-mini`: cambia qué combos legales banea, pero no cuánto acierta.

## 1. Sobre cómo medir

### La exactitud de 3 clases engañaba

La métrica original era la exactitud de 3 clases: qué fracción de las veces `risk` coincide con la etiqueta. Al mirar la matriz de confusión de una corrida apareció esto:

| Esperado \ juez | ok | watch | likely_ban |
| --- | --- | --- | --- |
| ok | 5 | **9** | 0 |
| watch | 0 | 6 | 3 |
| likely_ban | 0 | **8** | 6 |

El juez **nunca se equivocó por dos escalones**: no dijo `likely_ban` para una carta sana, ni `ok` para una baneada. Todos sus errores fueron a la clase vecina, casi siempre a `watch`. El problema es que `watch` no es un dato: una carta está baneada o no, y "vigilar" es un criterio propio del gold set. La métrica medía si el juez pone el corte en el mismo lugar que nosotros.

Además, **no había contra qué comparar**. Responder siempre `likely_ban` daba 0,38, y la mejor config daba 0,46, con un intervalo de confianza de 0,31 a 0,62.

**Cambio:** la métrica principal pasó a ser la **exactitud balanceada binaria**. El veredicto se lee como baneada (`likely_ban`) o no (`ok` o `watch`), y se promedian dos tasas: bans detectados y cartas legales respetadas. Cualquier respuesta constante saca exactamente 0,5, así que la línea base viene incluida. El juez sigue diciendo `watch`, porque al usuario le sirve, pero la métrica principal no depende de ese corte.

### Ban por texto y ban por metagame

Al revisar las cartas baneadas del gold set vimos que eran de dos tipos:

- **Rotas por su texto:** leerlas, solas o junto con otra carta, alcanza para ver el problema. Por ejemplo, Dark Depths con Vampire Hexmage crea una criatura 20/20 voladora e indestructible en el turno 3.
- **Baneadas por metagame:** su texto es razonable, pero en la práctica dominaron los torneos. Up the Beanstalk, en Standard, es "robá una carta cuando lanzás un hechizo de 5 o más de maná"; se baneó porque el mazo que la usaba ganaba demasiado.

Lo segundo depende de qué otras cartas y mazos existen, información que el sistema no tiene. De las 14 cartas baneadas del gold set original, 9 eran por metagame, incluidas las 4 de Standard. El sistema estaba siendo penalizado por algo imposible.

**Cambio:** cada ban tiene un `ban_reason` (`text` o `meta`). La métrica principal usa solo los bans por texto, y los de metagame se reportan aparte, como dato informativo.

### El gold set era chico y demasiado estricto

- **Pocos bans por texto.** Con 5, el intervalo de confianza de la detección iba de 0,12 a 0,77: ninguna comparación significaba nada. Agregamos 4 cartas baneadas por su texto (Mycosynth Lattice, Hypergenesis, Blazing Shoal y Skullclamp) y el gold set pasó a 41 cartas. Con 9, cada carta mueve la detección 0,11.
- **Una sola carta correcta por rol.** El gold set pedía una pieza puntual (Pestermite para Splinter Twin), cuando una docena de cartas cumplen el mismo rol. Pasamos a aceptar cualquier carta del rol: salidas de sacrificio, rituales de maná, efectos de "volver a jugar", etc.
- **Sinergias de mazo que no se deducen del texto** (Bloodghast con Hogaak). Quedaron marcadas como `synergy_pieces` y no se puntúan.

### El juez puede reconocer cartas famosas

Si el prompt dice "Splinter Twin", el modelo puede responder de memoria que está baneada. Por eso las cartas se envían como "Test Card". Pero no alcanza: un texto como "Add {R}{R}{R}{R}{R}." identifica a Seething Song igual. Un modelo más fuerte (`gpt-5.5`) llegó a escribir literalmente "Sacrifice Lotus Bloom" al describir la pieza de un combo. Esto infla los resultados en el gold set frente a cartas nuevas y no está medido todavía.

### Encontrar la pieza no es lo mismo que mostrarla

El recall mide si la búsqueda trae la pieza de combo. Pero en una prueba la búsqueda trajo Sanguine Bond en el puesto 2 para Exquisite Blood, y el juez igual no vio el combo. **Recall 1, pieza en el reporte 0.** Agregamos la métrica "piezas nombradas en el reporte" para medir lo que realmente le llega al usuario. En la mejor corrida anterior solo el 6 % de las piezas llegaba al reporte.

### Faltaba la línea base sin RAG

La config llamada `naive` también buscaba. No había forma de saber si el RAG ayudaba al veredicto o si el LLM lo hacía solo. Agregamos `no_rag`: el mismo juez, sin búsqueda.

### Otras decisiones de método

- **Regla de decisión fijada antes de correr:** una config mejora a otra solo si la supera por 0,15 o más de exactitud balanceada. Lo decidimos antes de ver los números para no elegir la conclusión después.
- **Recalcular gratis:** las corridas guardan cada reporte, así que se pueden puntuar de nuevo con otras métricas sin llamar a la API (`--rescore`). Con poco presupuesto esto fue clave.
- **Prueba con una carta antes de cada corrida completa**, para no perder plata con un error a mitad de camino.

## 2. Sobre el juez

### Calibrar el prompt rindió más que mejorar la búsqueda

La primera versión del prompt nombraba `ok`, `watch` y `likely_ban` sin definirlos, y el juez respondía `watch` para casi todo. Se agregaron dos cosas:

- **Criterios explícitos** para cada nivel. Por ejemplo, `likely_ban` es mucho más fuerte que todo lo comparable a su costo, o habilita un combo barato que gana o genera recursos infinitos. Y "la mayoría de las cartas son `ok`".
- **Justificación antes del veredicto:** un campo `risk_rationale` que el juez completa antes de elegir el nivel.

Con eso, la exactitud de `naive`, `pipeline` y `1_decompose` pasó de ~0,37 a ~0,49–0,51, y las falsas alarmas bajaron entre 0,14 y 0,21. **Fue una mejora mayor que la de cualquier cambio en la búsqueda, y costó menos de US$0,00005 por carta.**

### El juez es muy ruidoso

`pipeline` se corrió 3 veces sin cambiar nada. La búsqueda es determinista, porque el recall dio igual las tres veces. El juez no lo es, aunque la temperatura sea 0:

- 14 de las 37 cartas cambiaron de veredicto.
- La exactitud de 3 clases se movió entre 0,405 y 0,541; la exactitud balanceada, entre 0,57 y 0,69.

Conclusión: **una diferencia de menos de unas 3 cartas entre dos configs corridas una vez puede ser ruido.** De ahí la regla de 0,15.

### Sin contexto, el juez no se anima a banear

| Config | Exactitud balanceada | Bans por texto detectados | Legales marcadas como ban |
| --- | --- | --- | --- |
| Respuesta constante | 0,50 | — | — |
| `no_rag` (3 corridas) | 0,48 / 0,56 / 0,53 | 0, 1 y 1 de 9 | 1, 0 y 1 de 23 |
| `naive` | **0,76** | **5 / 9** | 1 / 23 |
| `4_rerank_llm` | 0,60 | 3 / 9 | 3 / 23 |

Sin búsqueda, el juez queda al nivel del azar. Con una búsqueda simple superó por 0,24 el promedio de `no_rag`, por encima del umbral fijado. **Pero al repetir `naive` dos veces más dio 0,57 y 0,60** (ver sección 5): con el promedio de tres corridas, 0,64, la mejora es de 0,12 y no llega al umbral.

Hay un matiz: el prompt le pide al juez comparar contra las cartas recuperadas y citar todo lo que afirma. Sin resultados no tiene con qué comparar ni qué citar, y casi nunca se anima a `likely_ban`. Lo que esto muestra es que **este juez necesita contexto para animarse a banear**, no que el modelo no sepa reconocer una carta rota. Una línea base del "LLM solo" justa usaría un prompt sin exigencia de citas.

### El criterio del prompt y las etiquetas no coinciden del todo

Cuando la búsqueda encontró el combo infinito de Kiki-Jiki con Splinter Twin, el juez dijo `likely_ban`, como manda el prompt para un combo barato e infinito. El gold set dice `watch`, porque Kiki-Jiki es legal. **Dos de las tres "legales marcadas como ban" de una config eran este desacuerdo, no un error del juez.** Hay que decidir si los combos de dos cartas son `likely_ban` o `watch` y alinear prompt y etiquetas.

### Los bans por metagame no se detectan

Las 4 cartas baneadas en Standard dieron `watch` en casi todas las configs. En sus justificaciones, el juez reconoce que son fuertes, pero no las considera rotas. Con solo el texto de la carta es probablemente un límite del enfoque, y la razón por la que las separamos.

## 3. Sobre la búsqueda

### La búsqueda trae cartas parecidas, no complementarias

Buscar con el texto de la carta trae variaciones de la misma carta: para Krark-Clan Ironworks, Krark-Clan Engineers. Pero una pieza de combo **complementa** a la carta, no se le parece: Pestermite no se parece a Splinter Twin. La mayoría de las piezas quedaban después del puesto 1000.

Para buscar lo complementario, el LLM descompone la carta y escribe cómo sería "la otra mitad del combo" (expansión de consultas). Ahí aparecieron los problemas siguientes.

### El modelo que escribe las contrapartes es lo que más importa

Piezas encontradas entre las 10 primeras, de 32:

| Modelo que descompone | Sin filtro | Con filtro de tipo y maná |
| --- | --- | --- |
| `gpt-4o-mini` | 2 | 0 |
| `gpt-4o` | 5 | 4 |
| `gpt-5.4-mini` con prompt reescrito | 7 | 6 |
| `gpt-5.5` | 16 | 21 |

- `gpt-4o-mini` repetía el efecto de la carta en vez de describir la otra mitad del combo, nombraba la carta evaluada y le ponía "encantamiento de 3 o menos" a casi todo.
- **El filtro por tipo y costo solo sirve si el modelo acierta el tipo.** Con `gpt-4o-mini` eliminaba piezas que la búsqueda sí había encontrado: Vizier of Remedies quedaba 1.ª sin filtro y desaparecía con filtro, porque el modelo había adivinado que era un encantamiento.
- `gpt-5.5` es mucho mejor, pero conoce los combos de memoria y es caro: una prueba parcial costó unos US$2.

### El embedding no entiende la jerga de Magic

- **Confunde "counters" (contadores) con "counter" (contrarrestar).** Buscando "remove all counters from target permanent", la búsqueda densa traía Negate y Cancel; BM25 encontraba a Vampire Hexmage en el puesto 1. **Por eso conviene la búsqueda híbrida.**
- **Una habilidad aislada se parece mucho más que la carta entera.** "Untap target creature" tiene similitud 0,62 contra la carta completa de Pestermite y 0,91 contra su habilidad sola. Indexar por habilidad podría ayudar; no se probó.
- **No ayudaron:** traer más candidatas (50 en vez de 10) ni indexar un texto "limpio" (sin nombre, costo ni texto de recordatorio).

### Anonimizar el nombre le sacó recall a la búsqueda simple

En la evaluación anterior, `naive` tenía recall@10 de 0,105; con el nombre oculto, 0. El nombre de la carta probablemente ayudaba a encontrar sus piezas. Es lo esperable: con una carta nueva tampoco habría un nombre conocido.

## 4. Sobre qué ve el juez

### Más contexto empeora al juez

Fue la señal más consistente de las ablaciones. Las configs que le pasan al juez todo lo recuperado (~11.000 tokens) son las peores. El control lo confirmó: la misma config con solo 5 cartas para el juez subió de 0,378 a 0,459 de exactitud de 3 clases.

### El rerank tenía que elegir entre dos cosas que el juez necesita

El juez hace dos trabajos: **comparar el poder** de la carta contra cartas parecidas y **buscar combos** con cartas complementarias. Pero recibía una sola lista mezclada:

- Las cartas parecidas (1 búsqueda) se fusionaban con las candidatas a combo (hasta 5 búsquedas), así que quedaban relegadas por cantidad.
- El rerank con LLM puntúa "qué tan probable es que esta carta haga combo". Las parecidas sacan 0 y quedan afuera de las 5 que ve el juez.
- El cross-encoder hace lo contrario: compara contra el texto de la carta y favorece a las parecidas. Por eso tenía las menores falsas alarmas.

Los efectos se ven carta por carta:

- **Krark-Clan Ironworks** quedó junto a piedras de maná comunes y el juez concluyó "comparable a otros artefactos de maná, manejable": `ok`, para una carta baneada por su texto.
- **Hogaak, Nadu y Cori-Steel Cutter** quedaron junto a cartas sin relación, y el juez perdió la referencia de poder: `watch` en vez de `likely_ban`.

**Cambio (`8_split_context`):** dos listas separadas de punta a punta. Las 5 cartas más parecidas, sin rerank, van al juez bajo el título "cartas comparables"; las 5 mejores candidatas a combo, después del rerank, van bajo "candidatas a combo". El prompt le indica qué usar para cada cosa.

### Al juez le llegaban 60 reglas

Al probar el cambio anterior vimos que las configs con descomposición buscan reglas por el texto de la carta, por cada habilidad y por cada tema, y **todas** llegan al juez. Para Krark-Clan Ironworks fueron 60 reglas, casi todas irrelevantes: Vehículos, Fortificaciones, fichas de Comida, prioridad en combate… Eso explica buena parte del contexto de más: `naive` le pasa al juez 2.357 tokens por carta (10 reglas) y `4_rerank_llm`, 8.650.

**Cambio (`9_split_few_rules`):** la config 8 con a lo sumo 10 reglas para el juez. Correr las dos permite separar el efecto de cada cambio.

## 5. Resultados de los últimos cambios

Una corrida de cada config (la segunda se canceló para cuidar el presupuesto). Detalle por carta en `outputs/eval-v5/run-1/`.

| Config | Acierto baneada / no baneada | Bans por texto | Legales marcadas como ban | recall@10 | Tokens por carta | US$ |
| --- | --- | --- | --- | --- | --- | --- |
| `no_rag` (promedio de 3) | 0,52 | 0–1 / 9 | 0–1 / 23 | — | ~700 | 0,008 |
| `naive` | 0,76 | 5 / 9 | 1 / 23 | 0,00 | 2.357 | 0,020 |
| `4_rerank_llm` | 0,60 | 3 / 9 | 3 / 23 | 0,08 | 8.650 | 0,072 |
| `8_split_context` | 0,59 | 2 / 9 | 1 / 23 | 0,03 | 8.983 | 0,074 |
| `9_split_few_rules` | 0,48 | 2 / 9 | 6 / 23 | 0,05 | 4.784 | 0,048 |

- **Separar comparables y candidatas a combo no mejoró el veredicto.** La config 8 marca menos legales como ban que la 4 (1 contra 3), pero detecta un ban menos. Es un empate.
- **Recortar las reglas cambió hacia dónde se equivoca el juez, no cuánto acierta.** Con 10 reglas en vez de ~60, el juez lee la mitad de tokens y banea más, pero las cartas equivocadas. Detecta los mismos 2 bans por texto (Dark Depths y Mycosynth Lattice) y marca 6 legales como ban: Lightning Bolt, Birds of Paradise, Splinter Twin, Devoted Druid, Exquisite Blood y Mikaeus. Las reglas no eran lo que frenaba la detección.
- **La búsqueda sigue sin encontrar las piezas** (recall@10 entre 0,03 y 0,05). Todas las configs elaboradas parten de la idea "encontrar la contraparte para que el juez vea el combo", y ese primer paso no funciona. Al juez le llegan candidatas que no hacen combo, y eso lo confunde en una dirección u otra.
### `naive` repetida y la prueba con las piezas correctas

Para saber si el 0,76 de `naive` era real y si arreglar la búsqueda valía la pena, corrimos `naive` dos veces más y una config nueva, `oracle_pieces`. Es `naive` más las piezas de combo del gold set (la primera carta de cada rol), entregadas al juez como candidatas a combo, como si la búsqueda fuera perfecta. Detalle en `outputs/eval-v6/`.

| Config | Acierto baneada / no baneada | Bans por texto | Legales marcadas como ban | Piezas nombradas | US$ |
| --- | --- | --- | --- | --- | --- |
| `naive`, corrida 1 | 0,76 | 5 / 9 | 1 / 23 | 0,00 | 0,020 |
| `naive`, corrida 2 | 0,57 | 2 / 9 | 2 / 23 | 0,00 | 0,017 |
| `naive`, corrida 3 | 0,60 | 3 / 9 | 3 / 23 | 0,00 | 0,015 |
| **`naive`, promedio** | **0,64** | | | | |
| `oracle_pieces` | 0,58 | 3 / 9 | 4 / 23 | 0,72 | 0,020 |

- **El 0,76 de `naive` fue en parte suerte.** El promedio de tres corridas es 0,64. Contra `no_rag` (0,52) la diferencia es 0,12, por debajo del umbral de 0,15: con este gold set **no se puede afirmar que el RAG mejore el veredicto**. Todas las configs con búsqueda quedan entre 0,48 y 0,64, dentro del ruido.
- **Con las piezas correctas, el juez ve el combo pero no lo juzga bien.** Nombra las piezas en el 72 % de los casos (sin ellas, 0 %), así que las lee y las usa. Pero el acierto no sube:
  - Detecta Punishing Fire y Dark Depths, que con la búsqueda real casi nunca detecta.
  - Deja en `watch` a Hogaak, Nadu y Mycosynth Lattice, aun con su pieza delante.
  - Banea Thopter Foundry e Isochron Scepter, combos legales, justamente porque ahora ve su combo.
- **Lo que separa un combo baneado de uno legal no está en el texto.** Dark Depths + Thespian's Stage y Thopter Foundry + Sword of the Meek son, leídos, combos de dos cartas parecidos. Que uno esté baneado y el otro no depende de cuán rápido y consistente es en la práctica y de qué otros mazos hay, información que el juez no tiene. Nuestra separación entre ban "por texto" y "por metagame" era más borrosa de lo que creíamos.

**Conclusión:** el cuello de botella no es la búsqueda ni el armado del contexto, sino la decisión del juez. Mejorar el recall haría que el juez nombre los combos en el reporte (lo que sí le sirve al usuario), pero no que acierte más el veredicto.

## 6. Medir sin memoria

La sección anterior dejó tres preguntas abiertas: si la línea base sin RAG era justa, cuánto responde el juez de memoria, y si el problema del juez es dónde pone el corte. Las respondimos con cuatro configs nuevas y un set de cartas inventadas. Detalle en `outputs/eval-v7/` y `outputs/eval-v8/`.

### El LLM solo, sin exigencia de citas

`no_rag` tenía un problema: el prompt exige citar las cartas recuperadas, y sin búsqueda no hay nada que citar, así que el juez casi nunca se animaba a banear. `no_rag_free` es el mismo juez, sin búsqueda, con los mismos criterios de riesgo y sin las reglas de citas (`cite: false`).

| Config | Corridas | Promedio | Bans por texto | Legales marcadas como ban |
| --- | --- | --- | --- | --- |
| `no_rag` (con citas) | 0,48 / 0,56 / 0,53 | 0,52 | 0–1 / 9 | 0–1 / 23 |
| `naive` | 0,76 / 0,57 / 0,60 | 0,64 | 2–5 / 9 | 1–3 / 23 |
| `no_rag_free` | 0,71 / 0,67 / 0,69 | **0,69** | 5 / 9 las tres veces | 3–5 / 23 |

- **La exigencia de citas era lo que frenaba al juez sin RAG.** Sacarla sube el promedio 0,17, por encima del umbral.
- **El LLM solo queda al nivel de `naive`** (0,05 arriba, dentro del ruido), y es la config más estable que corrimos.
- **Inventa cartas.** Sin contar cuando se nombra a sí mismo, entre el 22 % y el 31 % de las cartas que pone en sus combos no existen o están baneadas en el formato. Para Dark Depths propuso Pillow Fort y Rude Awakening. `cite: catalog` descarta esos nombres del reporte, pero no puede cambiar el veredicto, que sale en la misma respuesta.

### Cuánto recuerda el modelo

`python -m banhammer.eval.memorization` le muestra al modelo cada carta como la ve el juez ("Test Card") y le pregunta cuál es. Reconoce **16 de 41**:

| Grupo | Reconocidas |
| --- | --- |
| Modern, comunes | 9 / 10 |
| Modern, bans por texto | 3 / 9 |
| Modern, combos legales | 3 / 9 |
| Modern, bans por metagame | 1 / 5 |
| Standard (todas) | 0 / 8 |

- **El modelo vio estas cartas en su entrenamiento.** Reconoce más las que más circulan en internet (Lightning Bolt, Counterspell) y ninguna de Standard, que son posteriores a su corte de datos. Es **contaminación de datos**: el conjunto de prueba estaba en los datos de entrenamiento.
- **Los 3 bans por texto que reconoce son justo los que `no_rag_free` detectaba nombrándolos** (Krark-Clan Ironworks, Dark Depths, Mycosynth Lattice). Sin ellos, detecta 2 de 6.
- **Anonimizar el nombre no alcanza:** "Marit Lage" delata a Dark Depths y "Add {R}{R}{R}{R}{R}" a Seething Song.
- **No toda la memoria es mala.** Conocer las cartas *existentes* es necesario para ver un combo con ellas. Lo que contamina es reconocer la *carta evaluada* y recordar su veredicto, algo que con una carta nueva no existe.

**Sin las cartas que reconoce.** Re-puntuando gratis las corridas guardadas solo con las 25 cartas que el modelo no reconoce (quedan 6 bans por texto y 11 legales):

| Config | Las 41 cartas | Las 25 no reconocidas |
| --- | --- | --- |
| `no_rag_free` (3 corridas) | 0,69 | **0,53** |
| `naive_hybrid` (2 corridas) | 0,59 | 0,54 |
| `naive_reasons` (1 corrida) | 0,60 | 0,49 |
| `oracle_pieces` con gpt-5.4-mini (1 corrida) | 0,62 | 0,58 |
| `no_rag_rule` (3 corridas) | 0,54 | 0,50 |

- **Toda la ventaja del LLM solo en el gold set era memoria:** sin las cartas que reconoce, queda en el nivel del azar.
- **Con cartas reales que el modelo no reconoce, ninguna config detecta bien.** Los 6 bans por texto que quedan son los difíciles: Hogaak, Punishing Fire, Nadu, Hypergenesis, Blazing Shoal y Skullclamp.
- Con tan pocas cartas, cada una mueve mucho el número. Las corridas de `naive` y `no_rag` (`outputs/eval-v4/`, `eval-v6/`) no estaban en la máquina donde se hizo este cálculo.

### Pares mínimos: cartas inventadas

`data/eval/pairs.jsonl` tiene 9 pares de cartas ficticias, 18 en total. Cada par tiene una versión rota (`likely_ban`) y otra justa (`ok`) que difieren en un detalle:

- 6 copian el mecanismo de un ban por texto del gold set (Skullclamp, Krark-Clan Ironworks, Hypergenesis, Blazing Shoal, Dark Depths, Hogaak). Por ejemplo, Gravebinder Harness da +2/-1 y roba dos cartas, con equipar {1}; Mourner's Harness da +2/+1 y roba una, con equipar {3}.
- 2 son de pura relación poder/costo: un 5/5 con prisa por {G} contra uno por {4}{G}.
- 1 es la trampa de Thopter Foundry: con Sword of the Meek, la activación gratis es un loop infinito y la que cuesta {1} no.

El modelo no reconoce ninguna de las 18. Tres corridas por config:

| Config | Promedio | Rotas detectadas | Justas marcadas como ban | Pares acertados enteros |
| --- | --- | --- | --- | --- |
| **`naive`** | **0,89** (0,89 / 0,83 / 0,94) | **9 / 9 las tres veces** | 1–3 / 9 | 6–8 de 9 |
| `no_rag_free` | 0,72 (0,67 / 0,72 / 0,78) | 5–6 / 9 | 1–2 / 9 | 4–5 de 9 |
| `no_rag_rule` | 0,59 (0,56 / 0,61 / 0,61) | 1–2 / 9 | 0 / 9 | 1–2 de 9 |

- **`naive` gana porque recupera el precedente.** Para cada versión rota que copia un ban, la búsqueda trae entre las 10 primeras la carta baneada original, marcada "(banned in modern)": Gravebinder Harness trae a Skullclamp; Frozen Abyss, a Dark Depths; Ossuary Colossus, a Hogaak. El juez aplica el criterio "casi una copia de una carta baneada".
- **No banea todo lo que se parece.** Mourner's Harness también trae a Skullclamp, y el juez la deja en `watch`.
- **Por qué esto no se veía en el gold set:** ahí la carta evaluada *es* la baneada, y `exclude_self` la saca de los resultados. El precedente más útil quedaba afuera por diseño.
- **El Advanced RAG no mejora al simple en los pares.** Una corrida de cada una (`outputs/eval-v12/`): `4_rerank_llm` saca 0,83 (8 de 9 rotas, 6 pares enteros) y `8_split_context` 0,72 (7 de 9, 4 pares). El rerank también recupera los precedentes y queda cerca de `naive`. Con `split_context`, al juez le llega la lista de candidatas a combo y el precedente pierde peso: deja en `watch` a Scrapforge Engine y banea Mourner's Harness y Ossuary Behemoth.
- **El LLM solo sostiene 0,72 sin memoria posible**, así que su criterio no es solo memoria. Pero sin precedentes se le escapan Gravebinder Harness y Kindled Fury.
- **El matiz:** 6 de los 9 pares copian un ban, así que el set favorece la búsqueda de precedentes. Un mecanismo roto sin precedente solo lo prueban los 2 pares de poder (que todas las configs aciertan) y el de Thopter. Ese par sigue sin resolverse: `naive` banea las dos versiones, porque no distingue un loop infinito de uno que cuesta maná.

### Rasgos concretos y una regla fija

Para separar lo que el juez percibe de lo que decide, `no_rag_rule` (`decide: rule`) le pide estimar rasgos concretos: qué logra el combo (gana, recursos infinitos o ventaja incremental), cuántas cartas necesita, el turno más temprano en que gana y el poder frente a su costo. La regla fija, definida antes de correr, aplica la política de Modern de que ningún mazo gane antes del turno 4: es `likely_ban` un combo de dos cartas o menos que gana o hace loop antes del turno 4, o una carta muy por encima de la curva sin desventaja real.

| Set | Corridas | Bans detectados | Legales marcadas como ban |
| --- | --- | --- | --- |
| Gold set | 0,56 / 0,50 / 0,56 | 0–1 / 9 | 0 / 23 |
| Pares | 0,56 / 0,61 / 0,61 | 1–2 / 9 | 0 / 9 |

- **Nunca banea una carta legal, pero casi no detecta nada.** La regla es razonable; el modelo estima mal los rasgos. Para Dark Depths propuso un combo de tres cartas con Archaeomancer y Ghostly Flicker en lugar del de dos con Vampire Hexmage; para Gearwing Workshop usó a Krark-Clan Ironworks, que está baneada.
- **El problema no es dónde pone el corte el juez, sino que no razona bien la mecánica.** Esto limita lo que podría aportar un fine-tuning que solo le enseñe el criterio.

### RAG con memoria, y un juez más capaz

`naive_hybrid` (`cite: hybrid`) es `naive` con un juez que, además de citar lo recuperado, puede nombrar cartas y reglas de memoria. Las cartas que nombra se validan contra el catálogo (tienen que existir y ser legales en el formato). La idea era sumar los precedentes de `naive` y el conocimiento de `no_rag_free`. Para gastar poco, se corrió una o dos veces.

| Config | Juez | Gold set | Pares | Pares acertados enteros | US$ (pares) |
| --- | --- | --- | --- | --- | --- |
| `naive` (3 corridas) | gpt-4o-mini | 0,64 | 0,89 (0,83–0,94) | 6–8 de 9 | ~0,005 |
| `naive_hybrid` (2 corridas) | gpt-4o-mini | 0,66 / 0,52 | 0,89 / 0,89 | 8 y 7 de 9 | ~0,007 |
| `naive_hybrid_gpt5_mini` (2 corridas) | gpt-5.4-mini | — | 1,00 / 0,83 (promedio 0,92) | 9 y 7 de 9 | ~0,08 |

- **Mezclar RAG y memoria no suma con el modelo chico.** El híbrido con `gpt-4o-mini` queda igual que `naive` en los pares y por debajo en el gold set (promedio 0,59). En la primera corrida acertó el par de Thopter, pero en la segunda baneó las dos versiones otra vez: fue suerte.
- **Un juez más capaz va en la dirección correcta, pero este set no puede medirlo.** `gpt-5.4-mini` (`reasoning_effort: low`) acertó los 9 pares en la primera corrida y 7 en la segunda. En promedio acierta un par más y marca una carta justa menos que `naive` con `gpt-4o-mini`, y saca 0,92 contra 0,89. Pero con `naive` en 0,89 la mejora máxima posible es 0,11, menos que el umbral: **los pares son demasiado fáciles para el RAG con precedentes y están cerca del techo.** Además, la primera corrida exageró, con el mismo patrón que `naive` (0,76 y después 0,57).
- **Lo que sí mejora de forma consistente es el par de Dark Depths.** En las dos corridas separa bien: Frozen Abyss es "casi una copia de una carta baneada en Modern, pero bastante más rápida que ese precedente" → `likely_ban`; Thawing Abyss es "bastante más lenta y menos opresiva" → `watch`. `naive` con `gpt-4o-mini` baneaba la versión justa las tres veces.
- **El par de Thopter sigue sin resolverse.** En la primera corrida acertó ("un Thopter Foundry más barato, que con Sword of the Meek arma un loop infinito determinista"). En la segunda lo invirtió: dijo que la versión gratis "no crea recursos infinitos" y que la que cuesta {1} "habilita un loop infinito barato". Además baneó Mourner's Harness inventando una "interacción de robo infinito". **Distinguir un loop infinito de uno que cuesta un recurso por vuelta sigue fuera del alcance de los dos modelos.**
- **Con `oracle_pieces`, que sí tiene margen, tampoco mejora.** Le dimos a `gpt-5.4-mini` las piezas correctas del gold set, con el mismo prompt de citas estrictas que el `oracle_pieces` original. Una corrida (US$0,21, `outputs/eval-v10/`):

  | Juez | Exactitud balanceada | Bans por texto | Legales marcadas como ban | Piezas nombradas |
  | --- | --- | --- | --- | --- |
  | gpt-4o-mini | 0,58 | 3 / 9 | 4 / 23 | 0,72 |
  | gpt-5.4-mini | 0,62 | 3 / 9 | 2 / 23 | 0,74 |

  - Deja en `watch` a Thopter Foundry e Isochron Scepter, los combos legales que `gpt-4o-mini` baneaba porque veía su combo. Pero banea otros dos legales (Heliod y Saheeli Rai).
  - Es más prudente con los bans por texto: deja en `ok` a Krark-Clan Ironworks, Punishing Fire, Nadu y Blazing Shoal, aun con su pieza delante.
  - **Cambia qué errores comete, no cuántos.** Distinguir un combo baneado de uno legal sigue fuera de su alcance, lo que refuerza la conclusión de la sección 5: lo que los separa (velocidad, consistencia, metagame) no está en el texto.
- **Advertencias:** son dos corridas de 18 cartas; se apoya en los precedentes y en lo que sabe de las cartas originales; no se corrió sobre el gold set; y cuesta unas 13 veces más por carta que `gpt-4o-mini` (US$0,0047 contra US$0,00035).

### Los motivos de cada ban

El juez veía "(banned in modern)" pero no el porqué. `python -m banhammer.ingest.ban_announcements` descarga los 59 anuncios de bans que lista el sitemap de Wizards (desde 2014; quedan en caché en `data/raw/ban-announcements/`), junta los párrafos que nombran cada carta baneada en Modern o Standard, y `gpt-4o-mini` resume el motivo en una o dos oraciones. Hay motivo para **46 de las 65** cartas baneadas. Faltan sobre todo los bans de 2011 a 2013 (las tierras artefacto, Skullclamp, Hypergenesis, Blazing Shoal), cuyos anuncios no están en el sitemap. Algunos motivos tienen ruido: el de Golgari Grave-Troll describe una restricción en otro formato, y algunas fechas no corresponden al ban. Con `ban_reasons: true`, el juez lee el motivo junto a la marca: "(banned in modern: Hogaak fue baneado por su enorme impacto en el metagame…)". No hace falta reindexar.

| Config | Corridas | Gold set | Pares | Pares acertados enteros |
| --- | --- | --- | --- | --- |
| `naive` | 3 | 0,64 | 0,89 (0,83–0,94) | 6–8 de 9 |
| `naive_reasons` | 1 | 0,60 | 0,94 | 8 de 9 |

- **Los motivos no cambian el veredicto de forma medible.** En los pares queda en el rango de `naive`; en el gold set, apenas por debajo. Es una sola corrida.
- **En el gold set, el juez nunca lee el motivo de la carta que evalúa** (se excluye a sí misma), solo los de sus vecinas. Detecta Hogaak y Krark-Clan Ironworks, pero sigue dejando en `watch` a Mycosynth Lattice, Hypergenesis y Skullclamp, y banea combos legales (Devoted Druid, Isochron Scepter, Exquisite Blood).
- **Los motivos hablan del metagame de su momento** ("más Top 8 que cualquier otro mazo"). Para una carta nueva dicen qué tipo de carta preocupa a Wizards, pero no si esta carta en particular va a dominar. Para eso harían falta datos del metagame actual.

### Conclusión

- **El gold set de cartas reales subestimaba al RAG y sobreestimaba al LLM solo.** El LLM recuerda las cartas famosas, y el precedente de cada ban quedaba excluido por diseño.
- **Con cartas inventadas, el RAG simple es la mejor config (0,89 contra 0,72 del LLM solo)**, con una diferencia por encima del umbral. Su valor está en recuperar precedentes baneados. Toda la v2 apuntaba a encontrar piezas de combo, que era el problema equivocado.
- **Lo que sigue sin resolverse** es lo que mostró `oracle_pieces`: distinguir un combo baneado de uno legal. Un juez más capaz (`gpt-5.4-mini`) tampoco lo resuelve: en los pares va en la dirección correcta pero no hay margen para medirlo, y con `oracle_pieces` saca 0,62, igual que `gpt-4o-mini`. Cambia qué errores comete, no cuántos. Lo que separa un combo baneado de uno legal no está en el texto, y un modelo un escalón más capaz no lo suple.

## 7. Lo que queda abierto

- **Un juez de razonamiento más fuerte** (`gpt-5.5`, o `gpt-5.4-mini` con más `reasoning_effort`), medido sobre los pares y con `oracle_pieces`. `gpt-5.4-mini` con esfuerzo bajo no mejoró de forma medible.
- **Más pares mínimos, sobre todo de mecanismos sin precedente baneado**, revisados por alguien que juegue: los 9 actuales los escribimos nosotros, y las versiones justas de Thawing Abyss y Gearwing Atelier son discutibles.
- **Datos de metagame** (participación de cada mazo, resultados de torneos) para el juez. Los motivos de los bans ya están (`ban_reasons: true`) y no cambiaron el veredicto; falta revisarlos a mano y completar los bans de 2011 a 2013.
- **Alinear el criterio de combos del prompt con las etiquetas** (Kiki-Jiki: ¿`watch` o `likely_ban`?).
- **Indexar cada habilidad por separado**, la mejora de búsqueda más prometedora sin probar.
- **Un modelo mejor para la descomposición**, cuando haya presupuesto.
- **Más cartas baneadas por su texto** en el gold set, para que las diferencias entre configs se puedan detectar.
