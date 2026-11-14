# Modelo de riesgo de incendio (API)

`construir_modelo.py` arma el modelo que usa la API en `GET /prediction/risk` a partir de [`revision_red_v3`](../revision_red_v3/LEEME.md). Lo que sale es un JSON que va en `NasaSpaceAppsCitax-Api/src/prediction/risk-model.json`.

```
pip install pandas numpy pyarrow
python construir_modelo.py ../revision_red_v3 ../../NasaSpaceAppsCitax-Api/src/prediction/risk-model.json
```

## Qué calcula

Probabilidad de al menos un foco VIIRS NOAA-20 (vegetación, sin confianza baja) en cada casillero de 0,25° durante los próximos 30 días.

1. **Climatología**: para cada casillero y mes calendario, la fracción de años con algún foco en la serie armonizada (jul-2002 a dic-2025). En el tramo que tradujo la red v3 desde MODIS Aqua, el conteo es fraccionario y se toma como la media de una Poisson: P = 1 − e^(−focos). Cada casillero se suaviza hacia el promedio de su zona de 2° (con un peso de 2 años).
2. **Ajuste con focos recientes**: una regresión logística con el logit de la climatología, más log(1 + focos) en el casillero, en el anillo de 8 vecinos y en el anillo siguiente de 16. Se entrena con los meses observados por VIIRS (2012–2025) usando los focos del mes anterior, y con una climatología que deja afuera el propio año.

La API trae de FIRMS los focos de los últimos 5 días, los escala a 30 días y aplica la fórmula. Si FIRMS no responde, devuelve solo la climatología.

## Validación

Ajuste con 2012–2021 y prueba con 2022–2025, unos 205.000 casos casillero × mes, con una tasa base del 23,5 %:

| | Brier | AUC |
|---|---|---|
| Solo climatología | 0,1069 | 0,893 |
| Climatología + focos recientes | 0,0978 | 0,901 |

## Límites

- No están los pesos de la red v3, solo sus salidas. La red entra a través de la serie armonizada, no se vuelve a correr.
- El modelo se ajustó con focos del mes anterior y la API usa una ventana de 5 días escalada. Es una aproximación.
- En las zonas con `confiabilidad` baja o "pocos datos", la climatología anterior a 2012 es menos firme. La API devuelve ese dato por casillero.
