# Evaluación: ¿el RAG mejora el veredicto?

Corrida del 2026-11-16 sobre el gold set de 41 cartas (`data/eval/gold.jsonl`), con las métricas y la regla de decisión de [el plan de evaluación](plan-evaluacion.md). El detalle por carta está en `outputs/eval-v4/`.

## Resultados

| Config | Acierto baneada / no baneada | Bans por texto detectados | Legales marcadas como ban | Bans por metagame detectados | recall@10 | US$ (41 cartas) |
| --- | --- | --- | --- | --- | --- | --- |
| Respuesta constante | 0,50 | — | — | — | — | 0 |
| `no_rag`, corrida 1 | 0,48 | 0 / 9 | 1 / 23 | 1 / 9 | — | 0,008 |
| `no_rag`, corrida 2 | 0,56 | 1 / 9 | 0 / 23 | 2 / 9 | — | 0,008 |
| `no_rag`, corrida 3 | 0,53 | 1 / 9 | 1 / 23 | 1 / 9 | — | 0,008 |
| **`naive`** | **0,76** | **5 / 9** (0,27–0,81) | 1 / 23 | 3 / 9 | 0,00 | 0,020 |
| `4_rerank_llm` | 0,60 | 3 / 9 (0,12–0,65) | 3 / 23 | 2 / 9 | 0,08 | 0,072 |

Entre paréntesis, el intervalo de confianza del 95 %. Costo total de las cinco corridas: US$0,12.

## Qué dicen

1. **Sin búsqueda, el juez no detecta bans.** `no_rag` detecta 0 o 1 de 9 bans por texto en las tres corridas y queda en 0,48–0,56, lo mismo que responder siempre lo mismo. Es muy prudente: casi nunca banea una carta legal, pero tampoco una baneada.
2. **Con búsqueda, el veredicto mejora.** `naive` saca 0,76, 0,24 más que el promedio de `no_rag` (0,52). Supera el umbral de 0,15 fijado antes de correr, así que cuenta como mejora.
3. **La búsqueda más elaborada no mejoró el veredicto.** `4_rerank_llm` saca 0,60, 0,16 menos que `naive`. Está apenas por encima del umbral y es una sola corrida, así que la conclusión prudente es que no es mejor, aunque cuesta 3,6 veces más. Coincide con la evaluación anterior: más contexto no ayuda al juez.
4. **Los bans por metagame casi no se detectan** (1 a 3 de 9), como se esperaba.

## Lo que hay que aclarar al presentarlo

- **`naive` mejora el veredicto sin encontrar ninguna pieza de combo** (recall@10 = 0). Trae cartas parecidas, no complementarias. Tampoco muestra cartas baneadas como precedente: ninguna de sus 10 cartas está baneada en ninguno de los bans por texto. En sus justificaciones, el juez razona sobre el texto de la carta ("puede generar maná infinito", "crea una criatura poderosa sin costo de maná").
- **La comparación con `no_rag` no mide "qué sabe el LLM solo".** El prompt del juez le pide comparar contra las cartas recuperadas, citar todo lo que afirma y no nombrar cartas que no estén en los resultados. Sin resultados, nunca puede cumplir los criterios de `likely_ban`. Lo que muestra esta comparación es que **este juez necesita contexto para animarse a banear**, no que el LLM no sepa reconocer cartas rotas. Una línea base justa en ese sentido usaría un prompt sin exigencia de citas.
- **El ruido es grande.** `no_rag` repetido tres veces varía 0,08, y la evaluación anterior midió 0,12 de variación en `pipeline`. `naive` y `4_rerank_llm` se corrieron una sola vez.
- **Con el nombre anonimizado, `naive` ya no encuentra piezas.** En la evaluación anterior, `naive` tenía recall@10 de 0,105. Esa corrida probablemente se hizo antes de anonimizar el nombre, y el nombre de la carta ayudaba a encontrar sus piezas. No se verificó.
