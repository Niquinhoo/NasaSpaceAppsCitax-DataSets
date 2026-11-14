# Archivos para la revisión de la red v3 (MODIS Aqua → VIIRS)

Todos los conteos son focos de calor de vegetación (type = 0), sin confianza baja, por casillero de 0,25° (unos 27 × 24 km) y mes calendario (fecha UTC). Fuente: NASA FIRMS, resúmenes anuales por país, procesamiento estándar.

## 1. Predicciones de la red v3 contra lo observado

| Archivo | Nivel |
|---|---|
| `predicciones_v3_celda_mes.parquet` | casillero 0,25° × mes (562.848 filas) |
| `predicciones_v3_zona2_mes.csv` | zona de 2° (unos 220 × 200 km) × mes |
| `predicciones_v3_pais_mes.csv` | total del país × mes |

Columnas: `lat_c`, `lon_c` (centro del casillero) o `zona_lat`, `zona_lon` (esquina suroeste de la zona), `mes`, `condicion`, `aqua` (focos de MODIS Aqua, la entrada), `obs_snpp` (focos VIIRS S-NPP observados), `obs_n20` (focos VIIRS NOAA-20 observados, desde abril 2018), `pred_snpp` (predicción de la red en escala S-NPP), `pred_n20` (= `pred_snpp` × 0,98, el cociente NOAA-20/S-NPP medido en 2018–2021).

`condicion` indica qué sabía el modelo que predijo ese mes:

| Valor | Meses | Modelo |
|---|---|---|
| `fuera_de_muestra` | 2012–2014 | red entrenada con 2016 a jun-2021 (validación 2015) |
| `fuera_de_muestra` | 2020 a jun-2021 | red entrenada con 2012–2017 (validación 2018–2019) |
| `validacion` | 2018–2019 | no se entrenó con estos meses, pero se usaron para decidir cuándo parar el entrenamiento |
| `fuera_calibracion` | jul-2021 a 2025 | red entrenada con 2012–2017 aplicada cuando la órbita de Aqua ya se desvía (pasa hasta 75 min más tarde en 2025) |

Los años 2015–2017 no están porque son de entrenamiento para alguno de los dos modelos.

Las pasadas en que un satélite no mandó datos (por ejemplo, Aqua de día el 17-ago-2020) están excluidas de entrada y de observado en el mismo mes, así que `obs_snpp` puede diferir levemente de los totales crudos.

## 2. Serie armonizada y datos crudos

- `serie_armonizada_mensual.parquet`: casillero × mes, jul-2002 a dic-2025, en focos "VIIRS NOAA-20 equivalentes". Columna `fuente`: MODIS Aqua traducido por la red v3 (hasta ene-2012), VIIRS S-NPP observado × 0,98 (ene-2012 a mar-2018) o VIIRS NOAA-20 observado (desde abr-2018). También `confiabilidad` y `error_red` de la zona de 2° (solo en el tramo traducido) y `region_foco`.
- `crudo_mensual.parquet`: casillero × mes × sensor (`aqua`, `terra`, `snpp`, `n20`), sin traducir. `vegetacion` = type 0.
- `confiabilidad_zonas.csv`: error de la red por zona de 2° en abr-2018 a jun-2021 contra NOAA-20, y la discrepancia entre los dos VIIRS (`piso_viirs`) en la misma zona.

## 3. Cómo se calcula el error

Error = Σ |predicho − observado| / Σ observado, sumando primero dentro de cada zona y mes (WAPE). La discrepancia entre VIIRS es la misma cuenta usando S-NPP × 0,98 como "predicción" de NOAA-20.
