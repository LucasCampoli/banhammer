# Plan de evaluación

Cómo se mide Banhammer: qué pregunta responde la evaluación, con qué cartas, con qué métricas, contra qué líneas base y qué configuraciones se comparan. Se define antes de correr, para que los resultados no elijan la métrica.

## La pregunta

Banhammer lee una carta y dice si es `ok`, `watch` (vigilar) o `likely_ban` (probablemente baneable), citando cartas y reglas que recupera de un índice. La evaluación responde dos preguntas:

1. **¿El veredicto acierta si una carta está baneada o no?**
2. **¿La búsqueda (el RAG) mejora ese veredicto y encuentra las cartas con las que la carta hace combo?**

## El gold set

`data/eval/gold.jsonl`: 41 cartas reales cuya respuesta conocemos. Se envían al sistema como "Test Card", para que el modelo no responda por haber memorizado el nombre.

| Grupo | Formato | Esperado | Cartas |
| --- | --- | --- | --- |
| Baneadas por su texto | Modern | `likely_ban` | 9 |
| Baneadas por metagame | Modern | `likely_ban` | 5 |
| Baneadas por metagame | Standard | `likely_ban` | 4 |
| Combos legales | Modern | `watch` | 9 |
| Cartas comunes | Modern | `ok` | 10 |
| Cartas comunes | Standard | `ok` | 4 |

**Ban por texto y ban por metagame.** Una carta está rota por su texto cuando leerla, sola o junto con otra, alcanza para ver el problema (Dark Depths con Vampire Hexmage crea una 20/20 en el turno 3). Se banea por metagame cuando su texto es razonable pero en la práctica dominó los torneos (Up the Beanstalk). Lo segundo depende de qué otras cartas y mazos existen en ese momento, información que el sistema no tiene. Por eso los bans por metagame se reportan aparte y no entran en la métrica principal.

19 cartas tienen **piezas de combo** conocidas. Cuando varias cartas cumplen el mismo rol (por ejemplo, cualquier criatura que enderece algo al entrar sirve para Splinter Twin), el gold set acepta cualquiera de ellas.

## Métricas

### Principal: acierto baneada / no baneada

El veredicto se lee como binario: baneada (`likely_ban`) o no (`ok` o `watch`). La métrica es la **exactitud balanceada** (*balanced accuracy*): el promedio de dos tasas.

- **Bans por texto detectados**: de las 9 cartas baneadas por su texto, cuántas recibieron `likely_ban`.
- **Legales respetadas**: de las 23 cartas legales, cuántas *no* recibieron `likely_ban`.

Una respuesta constante (siempre `likely_ban`, o nunca) saca exactamente 0,5. Eso hace de línea base sin costo. 1,0 es perfecto.

**Por qué no exactitud de 3 clases.** `watch` no es un dato verificable: una carta está baneada o no, pero "vigilar" es un criterio del gold set. En una corrida anterior, todos los errores del juez fueron de un solo escalón, casi todos hacia `watch`. Con exactitud de 3 clases, el número medía dónde pone el corte el juez, no si detecta bans. Además, la mejor configuración daba 0,46 contra 0,38 de una respuesta constante, dentro del margen de error.

### Secundarias

| Métrica | Qué mide |
| --- | --- |
| Bans por texto detectados y legales marcadas como ban | Las dos mitades de la principal, con su intervalo de confianza del 95 % |
| Bans por metagame detectados | Informativa: no se espera que el sistema los detecte |
| Error ordinal | Cuántos escalones se equivoca en promedio (`watch` por `ok` es 1, `likely_ban` por `ok` es 2) |
| recall@10 y recall@25 | Qué fracción de las piezas de combo aparece entre las primeras 10 o 25 cartas recuperadas |
| Piezas nombradas en el reporte | Qué fracción de las piezas aparece en los combos que el reporte le muestra al usuario |
| Afirmaciones descartadas | Afirmaciones del reporte sin una cita válida: mide cuánto inventa el juez |
| Costo por carta | Dólares de LLM y embeddings |

El recall y las piezas nombradas miden cosas distintas. En una prueba, la búsqueda trajo la pieza (Sanguine Bond para Exquisite Blood) en el puesto 2, pero el juez no vio el combo. Recall 1, piezas nombradas 0.

## Líneas base

| Línea base | Config | Qué responde |
| --- | --- | --- |
| Respuesta constante | ninguna | 0,5 de exactitud balanceada, por construcción |
| **Sin RAG** | `configs/no_rag.yaml` | El juez lee solo la carta. ¿Cuánto sabe el LLM por sí mismo? |
| RAG ingenuo | `configs/naive.yaml` | Una búsqueda con el texto de la carta, sin filtro de formato |

La comparación central es **sin RAG contra la mejor config**. Si no hay diferencia, la búsqueda no está mejorando el veredicto, aunque mejore el recall.

## Configuraciones

Ablación incremental: cada fila agrega una técnica a la anterior, así la diferencia entre filas vecinas es el efecto de esa técnica.

| # | Config | Agrega |
| --- | --- | --- |
| 0 | `no_rag` | Línea base sin búsqueda |
| 1 | `naive` | Búsqueda densa con el texto de la carta |
| 2 | `pipeline` | Filtro por formato (solo cartas legales o baneadas en él) |
| 3 | `ablation/1_decompose` | El LLM descompone la carta en habilidades y escribe consultas por las contrapartes del combo |
| 4 | `ablation/2_hybrid` | BM25 además de la búsqueda densa, fusionados por rango |
| 5 | `ablation/3_type_mana_filter` | Filtra las contrapartes por tipo y costo adivinados |
| 6 | `ablation/4_rerank_llm` | Un LLM reordena 25 candidatas y el juez ve las 5 primeras |
| 7 | `ablation/5_rerank_cross_encoder` | Alternativa a la 6: reordena un cross-encoder local |

Quedan fuera de la corrida principal:

- `3b_context5` y `6_wide_rerank`: experimentos laterales. `6_wide_rerank` ya se corrió y no mejoró.
- `7_decompose_gpt5_mini`: es la que mejor recall dio en pruebas de búsqueda, pero cuesta unos US$0,60 por corrida completa. Solo se mide su recall (`--retrieval-only`) si sobra presupuesto.

## Cómo leer los resultados

- **Con 9 bans por texto, cada carta mueve la detección 0,11 y la exactitud balanceada 0,06.** Por ejemplo, 5 de 9 tiene un intervalo de 0,27 a 0,81.
- **Regla de decisión, fijada antes de correr:** una config mejora a otra solo si la supera por **0,15 o más** de exactitud balanceada (unas 3 cartas). Una diferencia menor se reporta como empate.
- Cada config corre una vez. El juez tiene `temperature 0`, pero las respuestas igual varían entre corridas. Si la mejor config y `no_rag` quedan cerca, se repiten las dos.

## Presupuesto

Quedan US$0,80. Costos medidos por carta: `no_rag` US$0,0002, `4_rerank_llm` US$0,0016.

| Corrida | Costo estimado |
| --- | --- |
| Configs 0 a 7, una vez cada una | ~US$0,30 |
| Repetir `no_rag` y la mejor config | ~US$0,10 |
| Margen | ~US$0,40 |

Las corridas guardadas se pueden puntuar de nuevo sin costo: `python -m banhammer.eval --rescore outputs/eval/<config>.json`.

## Limitaciones

- **El gold set es chico.** Las diferencias pequeñas no son evidencia; los intervalos de confianza lo muestran.
- **La memorización no se elimina del todo.** Se oculta el nombre, pero un texto famoso ("Add {R}{R}{R}{R}{R}.") se puede reconocer igual. Medir cuánto reconoce el modelo cuesta menos de un centavo y está pendiente.
- **Las etiquetas `watch` y las razones de cada ban son criterio propio**, no datos oficiales.
- **Algunas piezas son sinergias de mazo**, no combos deducibles del texto (Bloodghast con Hogaak). Esas no se puntúan.
