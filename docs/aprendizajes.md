# Lo que aprendimos evaluando Banhammer

Banhammer lee una carta de Magic: The Gathering y dice si es `ok`, `watch` (vigilar) o `likely_ban` (probablemente baneable). Para eso busca cartas y reglas relacionadas en un índice vectorial (RAG) y le pide a un LLM, el "juez", un reporte que cite lo que encontró.

Este documento junta lo que fuimos aprendiendo en las sucesivas evaluaciones: qué medíamos mal, qué descubrimos del juez, qué descubrimos de la búsqueda y qué cambiamos en consecuencia. Los detalles están en [la evaluación con el juez calibrado](resultados-eval-etapa-2-juez-v2.md), [el análisis de recall](resultados-analisis-recall.md), [el plan de evaluación](plan-evaluacion.md) y [la comparación contra la línea base sin RAG](resultados-eval-v4.md).

Todo se hizo con `gpt-4o-mini` y `text-embedding-3-small`, y con un presupuesto total de unos pocos dólares de API.

## Resumen

1. **La primera métrica medía el criterio del gold set, no el sistema.** La exactitud de 3 clases dependía de dónde se pone el corte de `watch`, que no es un dato verificable. La cambiamos por una métrica binaria (baneada o no) donde responder siempre lo mismo da 0,5.
2. **No todos los bans se pueden deducir del texto.** Muchas cartas se banean por cómo dominaron los torneos (metagame), no por lo que dicen. Las separamos de la métrica principal.
3. **El juez es muy ruidoso.** Con tres corridas idénticas, 14 de 37 cartas cambiaron de veredicto. Toda comparación de una sola corrida necesita una regla de decisión fijada de antemano.
4. **Calibrar el prompt del juez rindió más que cualquier mejora de la búsqueda**, y costó prácticamente nada.
5. **El RAG sí mejora el veredicto, pero el RAG más simple fue el mejor.** El juez sin búsqueda queda al nivel del azar (0,52); con una búsqueda simple sube a 0,76; con la búsqueda más elaborada, a 0,60.
6. **El contexto del juez estaba mal armado.** Una sola lista obligaba a elegir entre cartas comparables y candidatas a combo, y le llegaban unas 60 reglas, casi todas irrelevantes. Esto llevó a los dos últimos cambios.

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

Sin búsqueda, el juez queda al nivel del azar. Con una búsqueda simple supera por 0,24 el promedio de `no_rag`, por encima del umbral fijado.

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

*Pendiente: `8_split_context` y `9_split_few_rules`, dos corridas cada una.*

## 6. Lo que queda abierto

- **Medir la memorización:** preguntarle al modelo qué carta es cada texto anonimizado. Cuesta menos de un centavo.
- **Una línea base justa del LLM solo**, con un prompt sin exigencia de citas.
- **Alinear el criterio de combos del prompt con las etiquetas** (Kiki-Jiki: ¿`watch` o `likely_ban`?).
- **Indexar cada habilidad por separado**, la mejora de búsqueda más prometedora sin probar.
- **Un modelo mejor para la descomposición**, cuando haya presupuesto.
- **Más cartas baneadas por su texto** en el gold set, para que las diferencias entre configs se puedan detectar.
