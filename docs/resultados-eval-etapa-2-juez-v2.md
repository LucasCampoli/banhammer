# Evaluación de la etapa 2 con el juez calibrado

Corrida del 2026-11-15, sobre el mismo gold set (`data/eval/gold.jsonl`, 37 cartas) y las mismas métricas que [la primera evaluación](resultados-eval-etapa-2.md).
El detalle por carta está en `outputs/eval-v2/`: `noise-1`, `noise-2` y `noise-3` tienen las tres corridas de `pipeline`, y `all` tiene el resto de las configs.

## Qué cambió

La primera evaluación mostró que el juez respondía `watch` para casi todo, así que `risk_accuracy` no se movía según qué se recuperara. Para separar el problema del juez del de la búsqueda, se cambió solo el juez:

1. **Criterios de riesgo en el prompt** (`src/banhammer/pipeline/graph.py`). Antes el prompt nombraba `ok`, `watch` y `likely_ban` sin definirlos. Ahora dice:
   - `likely_ban`: la carta es mucho más fuerte que todo lo comparable a su costo, o habilita un combo barato y rápido que gana o genera recursos infinitos, o es casi una copia de una carta baneada en el formato.
   - `watch`: es pieza de un combo conocido que necesita varias cartas específicas, o está en el techo de poder para su costo pero tiene una desventaja real o una respuesta clara.
   - `ok`: su poder está en línea con las cartas comparables y no tiene combos peligrosos citables. La mayoría de las cartas son `ok`. Ser eficiente, popular o tener sinergias no alcanza para `watch`.

   No se agregaron ejemplos con cartas reales, para no filtrar el gold set al prompt.

2. **Justificación antes del veredicto** (`src/banhammer/models.py`). `BalanceReport` tiene un campo nuevo, `risk_rationale`, ubicado justo antes de `risk`. Obliga al juez a aplicar los criterios por escrito antes de elegir el nivel.

Además se agregaron dos corridas que no estaban en la primera evaluación:

- **`ablation/3b_context5`** (config nueva): `3_type_mana_filter` con el juez limitado a las 5 primeras cartas, sin rerank. `4_rerank_llm` cambiaba dos cosas a la vez (rerank y contexto de 5 cartas), y este control permite separar el efecto de cada una.
- **`ablation/5_rerank_cross_encoder`**: igual que `4_rerank_llm` pero con un reranker local (`cross-encoder/ms-marco-MiniLM-L-6-v2`). No se había corrido porque faltaba instalar `sentence-transformers` y torch.

## Ruido entre corridas

`pipeline` se corrió 3 veces sin cambiar nada:

| Corrida | risk_accuracy | false_alarm | Modern baneadas | Modern combos | Modern sanas | Standard baneadas | Standard sanas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0,514 | 0,357 | 0 / 7 / 3 | 0 / 7 / 2 | 7 / 3 / 0 | 0 / 4 / 0 | 2 / 2 / 0 |
| 2 | 0,405 | 0,429 | 0 / 6 / 4 | 0 / 3 / 6 | 6 / 4 / 0 | 0 / 4 / 0 | 2 / 2 / 0 |
| 3 | 0,541 | 0,357 | 0 / 5 / 5 | 0 / 6 / 3 | 6 / 4 / 0 | 0 / 4 / 0 | 3 / 1 / 0 |
| **Promedio** | **0,486** | **0,381** | | | | | |

Las celdas por grupo son `ok / watch / likely_ban`.

- **14 de las 37 cartas cambian de veredicto** entre las tres corridas.
- La búsqueda de `pipeline` es determinista (el recall es igual en las tres), así que **todo el ruido viene del juez**: `gpt-4o-mini` no es determinista aunque la temperatura sea 0.
- `risk_accuracy` se mueve entre 0,405 y 0,541 sin tocar nada, unas 5 cartas. **Cualquier diferencia de menos de unas 5 cartas entre dos configs corridas una sola vez puede ser ruido.** Esto vale también para las comparaciones de la primera evaluación.

## Resultados

Las configs que no son `pipeline` se corrieron una sola vez.

| Config | recall@5 | recall@10 | recall@25 | risk_accuracy | false_alarm | dropped/carta |
| --- | --- | --- | --- | --- | --- | --- |
| `naive` | 0,053 | 0,105 | — | 0,514 | 0,500 | 0,027 |
| `pipeline` (promedio de 3) | 0,053 | 0,105 | — | 0,486 | 0,381 | 0,018 |
| `1_decompose` | 0,053 | 0,132 | 0,132 | 0,486 | 0,429 | 0,054 |
| `2_hybrid` | 0,053 | 0,053 | 0,158 | 0,351 | 0,643 | 0,081 |
| `3_type_mana_filter` | 0,000 | 0,105 | 0,132 | 0,378 | 0,571 | 0,054 |
| `3b_context5` | 0,000 | 0,105 | 0,132 | 0,459 | 0,571 | 0,054 |
| `4_rerank_llm` | **0,132** | **0,132** | 0,132 | 0,405 | 0,786 | 0,027 |
| `5_rerank_cross_encoder` | 0,079 | 0,079 | 0,132 | 0,486 | **0,357** | 0,081 |

`naive` y `pipeline` no tienen recall@25 porque hacen una sola búsqueda con `k: 10`.

### Juez viejo contra juez nuevo

| Config | risk_accuracy (viejo → nuevo) | false_alarm (viejo → nuevo) |
| --- | --- | --- |
| `naive` | 0,351 → 0,514 | 0,643 → 0,500 |
| `pipeline` | 0,378 → 0,486 | 0,571 → 0,381 |
| `1_decompose` | 0,378 → 0,486 | 0,643 → 0,429 |
| `2_hybrid` | 0,378 → 0,351 | 0,786 → 0,643 |
| `3_type_mana_filter` | 0,324 → 0,378 | 0,786 → 0,571 |
| `4_rerank_llm` | 0,351 → 0,405 | 0,714 → 0,786 |

### Qué respondió el juez por grupo

Cada celda es `ok / watch / likely_ban`.

| Config | Modern baneadas (`likely_ban`) | Modern combos (`watch`) | Modern sanas (`ok`) | Standard baneadas (`likely_ban`) | Standard sanas (`ok`) |
| --- | --- | --- | --- | --- | --- |
| `naive` | 0 / 4 / 6 | 0 / 6 / 3 | 5 / 5 / 0 | 0 / 4 / 0 | 2 / 2 / 0 |
| `pipeline` (corrida 1) | 0 / 7 / 3 | 0 / 7 / 2 | 7 / 3 / 0 | 0 / 4 / 0 | 2 / 2 / 0 |
| `1_decompose` | 0 / 4 / 6 | 0 / 4 / 5 | 6 / 3 / 1 | 0 / 4 / 0 | 2 / 2 / 0 |
| `2_hybrid` | 0 / 6 / 4 | 0 / 4 / 5 | 5 / 5 / 0 | 0 / 4 / 0 | 0 / 4 / 0 |
| `3_type_mana_filter` | 0 / 7 / 3 | 0 / 5 / 4 | 5 / 5 / 0 | 0 / 4 / 0 | 1 / 3 / 0 |
| `3b_context5` | 0 / 6 / 4 | 0 / 7 / 2 | 4 / 6 / 0 | 0 / 4 / 0 | 2 / 2 / 0 |
| `4_rerank_llm` | 0 / 4 / 6 | 0 / 6 / 3 | 2 / 8 / 0 | 0 / 4 / 0 | 1 / 3 / 0 |
| `5_rerank_cross_encoder` | 0 / 6 / 4 | 0 / 5 / 4 | 6 / 4 / 0 | 1 / 3 / 0 | 3 / 1 / 0 |

Con el juez viejo, las baneadas de Modern recibían entre 1 y 3 `likely_ban` según la config. Con el nuevo reciben entre 3 y 6.

## Costo

| Config | Entrada/carta | Salida/carta | Embeddings/carta | US$ por carta | US$ total (37 cartas) |
| --- | --- | --- | --- | --- | --- |
| `naive` | 2.406 | 221 | 103 | 0,00039 | 0,0145 |
| `pipeline` (3 corridas) | 2.417 | 222 | 103 | 0,00033–0,00047 | 0,0417 |
| `1_decompose` | 9.015 | 532 | 294 | 0,00168 | 0,0621 |
| `2_hybrid` | 11.515 | 560 | 294 | 0,00207 | 0,0765 |
| `3_type_mana_filter` | 11.968 | 559 | 297 | 0,00214 | 0,0790 |
| `3b_context5` | 6.985 | 525 | 292 | 0,00137 | 0,0506 |
| `4_rerank_llm` | 8.828 | 757 | 299 | 0,00178 | 0,0659 |
| `5_rerank_cross_encoder` | 6.975 | 502 | 294 | 0,00135 | 0,0500 |
| **Total** | | | | | **0,440** |

- Los criterios agregan unos 270 tokens de entrada por carta, menos de US$0,00005.
- `pipeline` varía entre corridas por la caché de OpenAI: las corridas 2 y 3 tuvieron unos 2.300 tokens por carta en caché, la 1 solo 322.
- El cross-encoder corre en CPU y no gasta API. Su costo es el de la descomposición más el juez con 5 cartas.

## Conclusiones

1. **Calibrar el juez sirvió más que cualquier cambio en la búsqueda.** En `naive`, `pipeline` y `1_decompose`, `risk_accuracy` pasa de ~0,37 a ~0,49–0,51 y las falsas alarmas bajan entre 0,14 y 0,21. Es una mejora mayor que la que dio cualquier ablación de búsqueda en la primera evaluación, y costó prácticamente nada.

2. **El juez es muy ruidoso.** Con tres corridas idénticas, 14 cartas cambian de veredicto y `risk_accuracy` se mueve unas 5 cartas. Por eso ninguna config de esta tabla se puede rankear con seguridad contra otra con una sola corrida.

3. **Más cartas en el contexto empeoran al juez.** Es la señal más consistente: `2_hybrid` y `3_type_mana_filter` (~11k tokens de entrada) son las peores en `risk_accuracy` tanto con el juez viejo como con el nuevo. El control lo confirma: `3b_context5` es `3_type_mana_filter` con solo 5 cartas al juez, y sube de 0,378 a 0,459.

4. **El rerank con LLM mejora la búsqueda pero empeora al juez.** Comparado con `3b_context5`, que le pasa las mismas 5 cartas sin reordenar, sube recall@5 de 0 a 0,132: pone piezas de combo en el top 5. Pero las falsas alarmas suben a 0,786, las más altas de la tabla. Hipótesis, no verificada carta por carta: como el rerank busca cartas que hagan combo, incluso para una carta sana le muestra al juez cartas con potencial de combo, y el juez la marca.

5. **El cross-encoder es el mejor compromiso de las ablaciones.** Sube recall@5 a 0,079 (menos que el LLM, por una o dos cartas), tiene las menores falsas alarmas de la tabla (0,357, empatado con la mejor corrida de `pipeline`) y es la ablación más barata. Elige cartas de texto parecido en vez de cartas que hacen combo, y eso no dispara alarmas. Es también la primera config en que el juez se queda corto: marcó `ok` a una baneada de Standard.

6. **Las baneadas de Standard siguen sin detectarse.** Dan `watch` en todas las configs salvo una, que da `ok`. Según las justificaciones del juez, ve que son fuertes ("strong effect for its cost", "warrant close monitoring") pero no las considera rotas. Heartfire Hero, Up the Beanstalk, Proft's Eidetic Memory y Cori-Steel Cutter se banearon por su rendimiento en el meta, no porque el texto sea obviamente roto. Con solo el texto de la carta y `gpt-4o-mini`, probablemente sea un límite del enfoque y no algo que se arregle con el prompt.

## Recomendaciones

- **Config por defecto:** `pipeline` con el juez nuevo. Es la más barata (menos de US$0,0005 por carta) y está entre las mejores en calidad.
- **Si se quiere usar búsqueda avanzada:** la descomposición con contexto chico (`3b_context5` o `5_rerank_cross_encoder`) en vez de pasarle al juez todos los hits fusionados.
- **Antes de sacar conclusiones finas entre configs:** correr cada una 3 veces y promediar. Conviene paralelizar las cartas en `score_config` (`src/banhammer/eval/__main__.py`): hoy cada config tarda unos 10 minutos porque las 37 cartas y sus ~16 consultas corren en serie. Las 9 configs 3 veces costarían unos US$1,30.
- **Para reducir el ruido del juez:** probar un modelo más fuerte para el juez o votar entre varias respuestas (por ejemplo, la mayoría de 3).
