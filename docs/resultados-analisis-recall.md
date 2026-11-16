# Por qué el recall de combos da bajo

Análisis del 2026-11-16, sobre el gold set (`data/eval/gold.jsonl`) y el índice `banhammer-text-embedding-3-small`. Sigue a [la evaluación con el juez calibrado](resultados-eval-etapa-2-juez-v2.md), donde recall@25 no pasa de 0,158 en ninguna config.

Solo cuentan las 19 cartas del gold set que tienen piezas de combo (32 piezas en total, Pestermite figura dos veces). Para separar la búsqueda del juez, casi todas las mediciones de este informe son **solo de búsqueda**: se busca en Pinecone hasta el puesto 1000 y se anota en qué puesto aparece cada pieza. Eso es determinista y cuesta centavos.

## Resumen

1. Los datos están bien: las 32 piezas están indexadas, son legales en Modern y tienen su texto de Scryfall.
2. **La búsqueda base encuentra cartas parecidas, no complementarias.** La mayoría de las piezas quedan después del puesto 1000.
3. **`gpt-4o-mini` escribe mal las contrapartes**, y el filtro de tipo y maná descarta las pocas piezas que sí se encuentran.
4. **El embedding no entiende bien la jerga de Magic.** Ni siquiera una contraparte bien escrita garantiza encontrar la pieza.
5. Más candidatos (`card_k`) y embeber un texto más limpio no mejoraron el recall. **El modelo que escribe las contrapartes es lo que más mueve los números.**

## 1. La búsqueda base trae cartas parecidas

`pipeline` busca con el texto completo de la carta, nombre incluido. Los primeros resultados son variaciones de la misma carta:

| Carta | Primeros resultados | Puesto de las piezas |
| --- | --- | --- |
| Krark-Clan Ironworks | Krark-Clan Engineers, Krark-Clan Stoker | 110 y >1000 |
| Heliod, Sun-Crowned | Heliod, God of the Sun, … | >1000 |
| Splinter Twin | Flameshadow Conjuring, Twinflame | >1000 |

Una pieza de combo complementa a la carta, no se le parece: Pestermite no se parece a Splinter Twin. Solo 4 piezas quedan en el top 10, y son las que sí se parecen a su carta (Sanguine Bond para Exquisite Blood, Pyretic Ritual para Seething Song, Eldrazi Temple para Eye of Ugin).

## 2. Las contrapartes de la descomposición

Para cada pieza se toma el mejor puesto entre las 5 contrapartes de su carta, y se cuenta si entra en el top 10:

| Modelo y prompt | Sin filtro | Con filtro de tipo y maná |
| --- | --- | --- |
| `gpt-4o-mini`, prompt actual | 2 / 32 | 0 / 32 |
| `gpt-4o`, prompt actual | 5 / 32 | 4 / 32 |
| `gpt-4o-mini`, prompt reescrito | 3 / 32 | 1 / 32 |
| `gpt-5.4-mini`, prompt actual | 4 / 32 | 3 / 32 |
| `gpt-5.4-mini`, prompt reescrito | 7 / 32 | 6 / 32 |
| `gpt-5.5`, prompt actual | 16 / 32 | 21 / 32 |

Qué falla con `gpt-4o-mini`:

- **Repite el efecto de la carta en vez de describir la otra mitad del combo.** Para Splinter Twin propone efectos que copian criaturas, cuando lo que falta es una criatura que enderece algo al entrar (Pestermite).
- **Nombra la carta evaluada** ("remove an ice counter from Dark Depths").
- **Le pone `Enchantment` y `mv<=3` a casi todas las contrapartes.** Como el filtro confía en esos datos, saca a la pieza antes de buscar: Vizier of Remedies queda 1.ª sin filtro y desaparece con filtro, porque es criatura. Sanguine Bond también queda 1.ª y desaparece, porque cuesta 5. Esto explica que `3_type_mana_filter` tenga recall@5 = 0.

`gpt-5.5` es mucho mejor y con él el filtro ayuda, porque acierta el tipo. Pero hay que tomarlo con cuidado:

- **Conoce estos combos de memoria:** en una contraparte escribió literalmente "Sacrifice Lotus Bloom". Con una carta nueva rendiría menos que en el gold set.
- **Es caro:** US$5 por millón de tokens de entrada y US$30 por millón de salida, y razona con muchos tokens ocultos que se cobran como salida. Esta prueba se cortó a la mitad porque la cuenta pasó de ~US$1 a ~US$3.

## 3. Límites del embedding

Puesto de cada pieza buscando con descripciones correctas, escritas a mano:

| Query | Pieza | Búsqueda densa | BM25 |
| --- | --- | --- | --- |
| "Remove all counters from target permanent." | Vampire Hexmage | >1000 (trae Negate, Cancel…) | **1** |
| "When this creature enters, untap target creature." | Pestermite | 20 | 38 |
| "When this creature enters, you may tap or untap target permanent." | Pestermite | 3 | — |
| "Whenever you gain life, target opponent loses that much life." | Sanguine Bond | 1 | 1 |

- **Confunde "counters" (contadores) con "counter" (contrarrestar).**
- **Una descripción correcta describe una familia de cartas.** Hay decenas de cartas que enderezan algo al entrar, y el gold set pide una en particular. Pestermite solo sale arriba si la frase coincide casi palabra por palabra con su texto.
- **BM25 compensa justo donde falla el embedding:** cuando la contraparte usa las palabras literales de la carta. En `2_hybrid` no se notó porque las contrapartes de `gpt-4o-mini` casi nunca las usaban. Conviene mantener el híbrido.

## 4. Cambios probados que no mejoraron el recall

**Más candidatos (`ablation/6_wide_rerank`):** es `4_rerank_llm` sin el filtro de tipo, con 50 cartas por query y 100 candidatos al reranker.

| Config | recall@5 | recall@25 | risk_accuracy | false_alarm | US$ total |
| --- | --- | --- | --- | --- | --- |
| `4_rerank_llm` | 0,132 | 0,132 | 0,405 | 0,786 | 0,066 |
| `6_wide_rerank` | 0,079 | 0,105 | 0,459 | 0,643 | 0,103 |

Con las contrapartes de `gpt-4o-mini`, la mayoría de las piezas están después del puesto 1000, así que traer 50 en vez de 10 no las alcanza. La diferencia con la fila 4 es de una carta y puede ser variación de la descomposición.

**Embeber un texto más limpio:** se indexó en un namespace aparte el tipo y el texto de reglas, sin nombre, costo, fuerza/resistencia ni texto de recordatorio. Con la query base dio lo mismo (3 piezas en el top 10 en los dos índices). Con las contrapartes de `gpt-4o-mini` mejoró apenas (de 2 a 3 en el top 10). Con contrapartes bien escritas empeoró (Pestermite pasó del 20 al 111, Vizier del 6 al 16). Se descartó y se borró el namespace.

La mejora grande del texto limpio aparece solo con la habilidad aislada: la query "untap target creature" tiene coseno 0,62 contra la carta completa de Pestermite y 0,91 contra su habilidad sola. Indexar cada habilidad por separado podría ayudar, pero no se probó.

## Conclusiones

- El recall bajo **no es un problema de datos ni de k**. Es la combinación de buscar por similitud lo que es complementario, un modelo débil escribiendo las contrapartes y un filtro que confía en tipos mal adivinados.
- **Lo que más mueve el recall es el modelo de la descomposición.** `gpt-5.4-mini` con un prompt reescrito pasa de 2 a 7 piezas en el top 10. Cuesta ~US$0,014 por carta, unas 30 veces más que `gpt-4o-mini`, pero es una sola llamada por carta.
- **El filtro de tipo y maná solo sirve con un modelo que acierte el tipo.** Con `gpt-4o-mini` perjudica.
- **El gold set exige una carta puntual por rol** (Pestermite, cuando una docena de cartas enderezan al entrar), así que el recall subestima lo que encuentra la búsqueda. Además, varias piezas son sinergias de mazo que no se deducen del texto (Bloodghast con Hogaak, Shuko con Nadu).

## Qué no se midió

- Cada prueba de modelo corrió una sola vez. Con `temperature 0` igual hay variación entre corridas.
- BM25 sobre el texto limpio, y la indexación por habilidad.
- El efecto de un modelo GPT-5 sobre el eval completo (juez y risk_accuracy).
