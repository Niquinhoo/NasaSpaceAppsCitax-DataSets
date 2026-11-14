"""Construye el modelo de riesgo de incendio que usa la API a partir de la serie de la red v3.

Uso:  python construir_modelo.py <carpeta revision_red_v3> <archivo JSON de salida>

Requiere pandas, numpy y pyarrow. Ver LEEME.md en esta carpeta.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

K_SUAVIZADO = 2.0  # peso, en años, del promedio de la zona de 2° al suavizar cada casillero
ANIO_CORTE = 2022  # se ajusta con 2012–2021 y se evalúa con 2022–2025


def probabilidad_mes(focos: pd.Series, fuente: pd.Series) -> np.ndarray:
    """P(al menos un foco NOAA-20) del casillero en ese mes."""
    traducido = fuente.str.startswith('MODIS Aqua traducido').to_numpy()
    valores = focos.to_numpy()
    # La red da un conteo esperado (fraccionario): se lo trata como media de una Poisson.
    return np.where(traducido, 1 - np.exp(-valores), (valores > 0).astype(float))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ajustar_logistica(x: np.ndarray, y: np.ndarray, iteraciones=50):
    """Regresión logística por Newton-Raphson (con una leve penalización L2 para estabilidad)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(X.shape[1])
    for _ in range(iteraciones):
        p = 1 / (1 + np.exp(-X @ beta))
        gradiente = X.T @ (y - p) - 1e-6 * beta
        hessiana = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(len(beta))
        paso = np.linalg.solve(hessiana, gradiente)
        beta += paso
        if np.abs(paso).max() < 1e-8:
            break
    return beta


def auc(score, y):
    orden = np.argsort(score)
    rangos = np.empty(len(score))
    rangos[orden] = np.arange(1, len(score) + 1)
    positivos = y.sum()
    return (rangos[y == 1].sum() - positivos * (positivos + 1) / 2) / (positivos * (len(y) - positivos))


def main(carpeta: Path, salida: Path):
    serie = pd.read_parquet(carpeta / 'serie_armonizada_mensual.parquet')
    zonas = pd.read_csv(carpeta / 'confiabilidad_zonas.csv')

    serie['anio'] = serie.period.dt.year
    serie['mes'] = serie.period.dt.month
    serie['p'] = probabilidad_mes(serie.focos, serie.fuente)

    celdas = serie[['gy', 'gx', 'lat_c', 'lon_c', 'zona_lat', 'zona_lon']].drop_duplicates(['gy', 'gx'])
    celdas = celdas.sort_values(['gy', 'gx']).reset_index(drop=True)
    celdas['i'] = np.arange(len(celdas))
    serie = serie.merge(celdas[['gy', 'gx', 'i']], on=['gy', 'gx'])

    # Matriz casillero × período (las filas ausentes de la serie son meses sin focos).
    periodos = pd.period_range('2002-07', '2025-12', freq='M')
    indice_periodo = {p: n for n, p in enumerate(periodos)}
    serie['t'] = serie.period.dt.to_period('M').map(indice_periodo)
    P = np.zeros((len(celdas), len(periodos)))
    F = np.zeros_like(P)
    P[serie.i, serie.t] = serie.p
    F[serie.i, serie.t] = serie.focos
    mes_de = np.array([p.month for p in periodos])
    anio_de = np.array([p.year for p in periodos])

    # Probabilidad climatológica por casillero y mes calendario, suavizada hacia su zona de 2°.
    zona_id = celdas.groupby(['zona_lat', 'zona_lon']).ngroup().to_numpy()
    suma = np.zeros((len(celdas), 12))
    n = np.zeros(12)
    for m in range(1, 13):
        columnas = mes_de == m
        suma[:, m - 1] = P[:, columnas].sum(axis=1)
        n[m - 1] = columnas.sum()
    media_zona = np.zeros_like(suma)
    for z in np.unique(zona_id):
        media_zona[zona_id == z] = suma[zona_id == z].mean(axis=0) / n
    prior = (suma + K_SUAVIZADO * media_zona) / (n + K_SUAVIZADO)

    # Vecindario: anillo 1 (8 casilleros) y anillo 2 (16 casilleros) en la grilla de 0,25°.
    pos = {(gy, gx): i for gy, gx, i in celdas[['gy', 'gx', 'i']].itertuples(index=False)}

    def vecinos(radio_min, radio_max):
        filas, columnas = [], []
        for (gy, gx), i in pos.items():
            for dy in range(-radio_max, radio_max + 1):
                for dx in range(-radio_max, radio_max + 1):
                    if radio_min <= max(abs(dy), abs(dx)) <= radio_max and (gy + dy, gx + dx) in pos:
                        filas.append(i)
                        columnas.append(pos[(gy + dy, gx + dx)])
        return np.array(filas), np.array(columnas)

    def suma_vecinos(matriz, par):
        filas, columnas = par
        resultado = np.zeros_like(matriz)
        np.add.at(resultado, filas, matriz[columnas])
        return resultado

    anillo1 = suma_vecinos(F, vecinos(1, 1))
    anillo2 = suma_vecinos(F, vecinos(2, 2))

    # Ejemplos: para cada mes observado por VIIRS (2012+), ¿hubo foco en el casillero
    # dado la climatología del mes y los focos del mes anterior en el casillero y alrededor?
    filas_x, filas_y, filas_prior, filas_anio = [], [], [], []
    for t in range(1, len(periodos)):
        if anio_de[t] < 2012:
            continue
        m = mes_de[t] - 1
        anio = anio_de[t]
        # Climatología sin el propio año, para no filtrar el resultado en la entrada.
        p_loo = (suma[:, m] - P[:, t] + K_SUAVIZADO * media_zona[:, m]) / (n[m] - 1 + K_SUAVIZADO)
        filas_x.append(np.column_stack([
            logit(p_loo), np.log1p(F[:, t - 1]), np.log1p(anillo1[:, t - 1]), np.log1p(anillo2[:, t - 1]),
        ]))
        filas_y.append((F[:, t] > 0).astype(float))
        filas_prior.append(p_loo)
        filas_anio.append(np.full(len(celdas), anio))
    X = np.vstack(filas_x)
    y = np.concatenate(filas_y)
    prior_ej = np.concatenate(filas_prior)
    anios = np.concatenate(filas_anio)

    entrenamiento = anios < ANIO_CORTE
    beta = ajustar_logistica(X[entrenamiento], y[entrenamiento])
    prueba = ~entrenamiento
    p_modelo = 1 / (1 + np.exp(-np.column_stack([np.ones(prueba.sum()), X[prueba]]) @ beta))
    metricas = {
        'testPeriod': f'{ANIO_CORTE}-01/2025-12',
        'testCases': int(prueba.sum()),
        'baseRate': round(float(y[prueba].mean()), 4),
        'brierClimatology': round(float(np.mean((prior_ej[prueba] - y[prueba]) ** 2)), 4),
        'brierModel': round(float(np.mean((p_modelo - y[prueba]) ** 2)), 4),
        'aucClimatology': round(float(auc(prior_ej[prueba], y[prueba])), 4),
        'aucModel': round(float(auc(p_modelo, y[prueba])), 4),
    }
    # Coeficientes finales con todos los años.
    beta = ajustar_logistica(X, y)

    zonas_lista = [
        {'lat': z.zona_lat, 'lon': z.zona_lon, 'reliability': z.confiabilidad,
         'networkError': None if pd.isna(z.error_red) else z.error_red}
        for z in zonas.itertuples(index=False)
    ]
    indice_zona = {(z['lat'], z['lon']): n for n, z in enumerate(zonas_lista)}
    # Las claves en inglés son las que lee la API.
    modelo = {
        'description': 'Riesgo de al menos un foco VIIRS NOAA-20 (vegetación, sin confianza baja) por casillero de 0,25° '
                       'en los próximos 30 días. Climatología de la serie armonizada de la red v3 (jul-2002 a dic-2025) '
                       'combinada por regresión logística con los focos recientes del casillero y su vecindario.',
        'cellDegrees': 0.25,
        'coefficients': dict(zip(['intercept', 'climatologyLogit', 'cellFocos', 'ring1Focos', 'ring2Focos'],
                                 np.round(beta, 5).tolist())),
        'metrics': metricas,
        'zones': zonas_lista,
        # [lat_c, lon_c, índice de zona (-1 si no tiene), prob. climatológica ene … dic]
        'cells': [
            [row.lat_c, row.lon_c, indice_zona.get((row.zona_lat, row.zona_lon), -1),
             *np.round(prior[row.i], 4).tolist()]
            for row in celdas.itertuples(index=False)
        ],
    }
    salida.write_text(json.dumps(modelo, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(json.dumps({'coefficients': modelo['coefficients'], 'metrics': metricas,
                      'cells': len(celdas)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
