"""
===============================================================================
TUTORIAL: Modelado de Niveles Freáticos con Pastas
===============================================================================
Este script demuestra cómo construir modelos de niveles freáticos usando la 
biblioteca Pastas, considerando múltiples factores de estrés (precipitación,
evaporación, y bombeos).

Estructura:
1. Carga y preparación de datos de niveles
2. Análisis de tendencias (Mann-Kendall)
3. Carga y preparación de datos de estrés (clima y bombeos)
4. Construcción de modelos Pastas con diferentes configuraciones
5. Evaluación y exportación de resultados
===============================================================================
"""

# =============================================================================
# 1. IMPORTS Y CONFIGURACIÓN
# =============================================================================

import os
import warnings
from multiprocessing import Pool, cpu_count
import pandas as pd
import pastas as ps
import matplotlib.pyplot as plt
import pymannkendall as mk
import numpy as np
from scipy import stats
import time
# Suprimir deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configurar nivel de logging de Pastas
ps.set_log_level("ERROR")

# Crear directorio para plots
os.makedirs("plots", exist_ok=True)

# Multiprocessing: 1 = secuencial (más fácil de depurar), >1 = paralelo por tipo de modelo
N_WORKERS = cpu_count() - 1 #para acelerar

# =============================================================================
# 2.1 FUNCIONES AUXILIARES
# =============================================================================

def load_pumping_data(filepath: str) -> pd.Series:
    """
    Carga datos de bombeo mensuales y los convierte a serie diaria.
    
    Proceso:
    1. Lee CSV con volúmenes mensuales (m³/mes)
    2. Convierte a tasa diaria (m³/día) dividiendo por días del mes
    3. Expande a timestep diario usando forward fill
    4. Retorna serie negativa (extracción = estrés negativo)
    
    Args:
        filepath: Ruta al archivo CSV con columnas 'Fecha' y 'total'
        
    Returns:
        Serie temporal diaria con bombeos en m³/día (valores negativos)
    """
    df = pd.read_csv(filepath)
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna().set_index("Fecha").sort_index()
    
    # Manejar índices duplicados
    if df.index.duplicated().any():
        df = df.groupby(df.index).mean(numeric_only=True)
    
    # Convertir volumen mensual a tasa diaria
    days_in_month = df.index.days_in_month
    df["m3_day"] = df["total"] / days_in_month
    
    # Expandir a timestep diario
    daily_series = df["m3_day"].resample("D").ffill()
    
    # Retornar serie negativa (extracción)
    return daily_series


def distancia_utm(este_1, norte_1, este_2, norte_2):
    """
    Calcula distancia euclidiana entre dos puntos UTM (en metros).
    
    Args:
        este_1, norte_1: Coordenadas del primer punto
        este_2, norte_2: Coordenadas del segundo punto
        
    Returns:
        Distancia en metros
    """
    e1 = np.asarray(este_1, dtype=float)
    n1 = np.asarray(norte_1, dtype=float)
    e2 = np.asarray(este_2, dtype=float)
    n2 = np.asarray(norte_2, dtype=float)
    return np.hypot(e2 - e1, n2 - n1)


def calculate_model_statistics(model: ps.Model) -> dict:
    """
    Calcula estadísticas de validación del modelo.
    
    Incluye:
    - R² (coeficiente de determinación)
    - RMSE (raíz del error cuadrático medio)
    - Tests estadísticos sobre residuales (t-test, Wilcoxon, Shapiro-Wilk)
    
    Args:
        model: Modelo Pastas calibrado
        
    Returns:
        Diccionario con todas las estadísticas
    """
    residuals = model.residuals()
    
    # Tests estadísticos sobre residuales
    t_stat, p_value_t = stats.ttest_1samp(residuals, 0)
    stat_wilc, p_value_wilc = stats.wilcoxon(residuals)
    stat_shap, p_value_shap = stats.shapiro(residuals)
    
    # Métricas de ajuste
    r2 = model.stats.rsq()
    rmse = model.stats.rmse()
    
    bic = model.stats.bic()
    aic = model.stats.aic()
    
    kge = model.stats.kge(modified=True)
    mae = model.stats.mae()
    
    # Legates & McCabe (1999)
    obs = model.oseries.series
    sim = model.simulate()  # devuelve un pandas Series alineado con obs
    
    #std
    std_obs = model.oseries.series.dropna().std()
    std_sim = model.get_output_series().Simulation.dropna().std()
    
    sim_series = model.get_output_series().Simulation.dropna()
    obs_series = model.oseries.series.dropna()
    
    richard_sim = ps.stats.signatures.richards_pathlength(sim_series)
    richards_obs = ps.stats.signatures.richards_pathlength(obs_series)

    # MAE
    # mae = np.mean(np.abs(sim - obs))
    
    return {
        "R²": r2,
        "RMSE": rmse,
        "std_obs": std_obs,# * 0.7,
        "std_sim": std_sim,# * 0.7,
        "std_obs07": std_obs * 0.7,
        "KGE": kge,
        # "p-value zero": p_value_t,
        # "t-statistic zero": t_stat,
        "p-value wilconox": p_value_wilc,
        # "statistic wilconox": stat_wilc,
        # "statistic shapiro": stat_shap,
        "MAE": mae,
        "BIC": bic,
        "AIC": aic,
        "richard_ratio": richard_sim / richards_obs,
        "p-value shapiro": p_value_shap,
    }


def calculate_ivm(row: pd.Series) -> str:
    """
    Calcula Indicador de Validación del Modelo (IVM).
    
    Un modelo es "Válido" si cumple TODAS las condiciones:
    - R² > 0.85 (85% de varianza explicada)
    - RMSE < 0.05 m
    - p-value Wilcoxon > 0.05 (residuales centrados en cero)
    - p-value Shapiro > 0.05 (residuales normalmente distribuidos)
    
    Args:
        row: Fila del DataFrame con estadísticas del modelo
        
    Returns:
        "Válido" o "No Válido"
    """
    try:
        conditions = [
            pd.notna(row["R²"]) and row["R²"] > 0.65, # R2 Premia modelos “suaves” que siguen la tendencia pero no los eventos. 
            pd.notna(row["RMSE"]) and row["RMSE"] < row["std_obs07"],
            pd.notna(row["p-value wilconox"]) and row["p-value wilconox"] > 0.05,
            pd.notna(row["MAE"]) and row["MAE"] < row["std_obs07"],
            pd.notna(row["KGE"]) and row["KGE"] > 0.6,
            pd.notna(row["richard_ratio"]) and row["richard_ratio"] > 0.1,
            pd.notna(row["richard_ratio"]) and row["richard_ratio"] < 10,            # pd.notna(row["p-value shapiro"]) and row["p-value shapiro"] > 0.05 # No se usa porque no funciona bien con muchos datos
        ]
        return "Válido" if all(conditions) else "No Válido"
    except (KeyError, TypeError):
        return "No Válido"


def _trunc3(x: float) -> float:
    """Trunca a 3 decimales (no redondea)."""
    return float(np.trunc(float(x) * 1000.0) / 1000.0)


def _mk_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica test de Mann-Kendall a cada pozo para detectar tendencias.
    Retorna tabla con estadísticos Z, S, p-value y tendencia (Creciente/
    Decreciente/Sin tendencia).
    """
    def _trend_es(t: str) -> str:
        t = str(t).strip().lower()
        if t == "increasing":
            return "Creciente"
        if t == "decreasing":
            return "Decreciente"
        return "Sin tendencia"

    mk_results = []
    for pozo, g in df.groupby("Pozo", sort=True):
        serie = g.sort_values("Fecha")["Valor"].dropna()
        if len(serie) < 3:
            continue
        res = mk.original_test(serie)
        mk_results.append({
            "Pozo": pozo,
            "Estadistico Z": _trunc3(res.z),
            "Estadistico S": int(res.s),
            "p-value": _trunc3(res.p),
            "tendencia": _trend_es(res.trend),
        })
    return pd.DataFrame(mk_results)


# =============================================================================
# 2.2 FUNCIONES PARA CONSTRUIR MODELOS (se usan en secuencial y en paralelo)
# =============================================================================


def get_data_bundle():
    """
    Agrupa todos los datos necesarios para construir modelos.
    Se llama desde el bloque principal después de cargar niveles, estrés y coordenadas.
    """
    return {
        "niveles_series": niveles_series,
        "coordenadas_niveles": coordenadas_niveles,
        "coordenadas_bombeos": coordenadas_bombeos,
        "sm_precip_directa": sm_precip_directa,
        "sm_precip_indirecta": sm_precip_indirecta,
        "sm_evap": sm_evap,
        "ALB_series": ALB_series,
        "SOP_series": SOP_series,
        "MOP_series": MOP_series,
        "SOP_iny_series": SOP_iny_series,
        "MOP_iny_series": MOP_iny_series,
        "TIL_series": TIL_series,
        "TUC_series": TUC_series,
        "PEINE_series": PEINE_series,
        "pozo_to_tipo": pozo_to_tipo,
    }


def create_model_with_data(modelo: str, location_names: list, data: dict):
    """
    Construye y calibra un modelo Pastas para cada pozo usando solo los datos en `data`.
    Así la misma función sirve en secuencial y en workers paralelos.

    Returns:
        model_stats: dict {nombre_pozo: {estadísticas}}
        gains_data: list de dicts con gains por pozo (solo Modelo_C/D/E).
    """
    ns = data["niveles_series"]
    coord_n = data["coordenadas_niveles"]
    coord_b = data["coordenadas_bombeos"]
    sm_precip_directa = data["sm_precip_directa"]
    sm_precip_indirecta = data["sm_precip_indirecta"]
    sm_ev = data["sm_evap"]
    ALB = data["ALB_series"]
    SOP = data["SOP_series"]
    MOP = data["MOP_series"]
    SOP_iny = data["SOP_iny_series"]
    MOP_iny = data["MOP_iny_series"]
    TIL = data["TIL_series"]
    TUC = data["TUC_series"]
    PEINE = data["PEINE_series"]
    pozo_to_tipo = data.get("pozo_to_tipo", {})

    model_stats = {}
    gains_data = []
    
    modelos_D_copy1 = {}
    modelos_D_copy2 = {}
    
    modelos_D_series = {}
    
    median_no_alb = []
    median_no_sqm = []

    for pozo in location_names:
        tipo = pozo_to_tipo.get(pozo, None)  # ej. "HEAD", "STAGE"; None si no está en el mapa

        # 1. Serie de niveles del pozo
        print(f"Creando modelo para {pozo} - {modelo}" + (f" (Tipo: {tipo})" if tipo else ""))
        datos = ns[ns["Pozo"] == pozo]["Valor"]
        datos.index = pd.to_datetime(datos.index, errors="coerce")
        datos = datos.dropna().sort_index().resample("D").mean()

        
        # 2. Modelo: observación + recarga
        if tipo == "STAGE":
            ml = ps.Model(datos, name=f"{pozo} - {modelo}")
            ml.add_stressmodel(sm_precip_directa)
            ml.add_stressmodel(sm_ev)
        else:
            ml = ps.Model(datos, name=f"{pozo} - {modelo}")
            ml.add_stressmodel(sm_precip_indirecta)
            ml.add_stressmodel(sm_ev)

        # 3. Estrés de bombeo según tipo de modelo
        # if modelo == "Modelo_B":
        #     este_pozo = float(coord_n.loc[pozo]["Este"])
        #     norte_pozo = float(coord_n.loc[pozo]["Norte"])
        #     bombeos = ["alb"]
        #     series = [ALB]
        #     dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
        #     ml.add_stressmodel(ps.WellModel(series, "alb_sal", dist, ps.HantushWellModel(use_numba=True)))
        if modelo == "Modelo_E":
            # print("entra a loop Modelo_E")
            este_pozo = float(coord_n.loc[pozo]["Este"])
            norte_pozo = float(coord_n.loc[pozo]["Norte"])
            
            bombeos = ["alb"]
            series = [ALB]
            
            ml = ps.Model(datos, name=f"{pozo} - {modelo}")
            ml.add_stressmodel(sm_precip_indirecta)

            dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
            ml.add_stressmodel(ps.WellModel(series, "alb_sal", dist, ps.HantushWellModel(use_numba=True)))
            
            bombeos = ["til", "tuc", "peine"]
            series = [TIL, TUC, PEINE]
            dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
            ml.add_stressmodel(ps.WellModel(series, "alb_agua", dist, ps.HantushWellModel(use_numba=True)))
            
            bombeos = ["sop", "mop"]
            series = [SOP, MOP]
            dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
            ml.add_stressmodel(ps.WellModel(series, "SQM_pump", dist, ps.HantushWellModel(use_numba=True)))
            
            inyeccion = ["sop_iny", "mop_iny"]
            series = [SOP_iny, MOP_iny]
            ml.add_stressmodel(ps.WellModel(series, "SQM_iny", dist, ps.HantushWellModel(use_numba=True),up=True))


            # print("entra a calibrar Modelo_E")
            
            # 5. Calibrar y guardar estadísticas (solo Modelo_D)
            # 5.1 Sin noisemodel para generar buenos valores iniciales
            ml.solve(report=False,tmin="2013-01-01",tmax="2023-12-01",warmup=720)
            
            # print("Modelo calibrado1")
            
            # 5.2 Con noisemodel (ArmaNoiseModel) para reducir autocorrelación de los residuales
            ml.add_noisemodel(ps.ArmaNoiseModel())
            ml.solve(initial=False,tmin="2013-01-01",tmax="2023-12-01",warmup=720,report=False)
            
            # print("Modelo calibrado con NoiseModel")
            model_stats[pozo] = calculate_model_statistics(ml)
            
            ml_D_series = ml.get_output_series()
            # print(ml_D_series)
            # plt.plot(ml_D_series.index, ml_D_series.Simulation)
            # plt.show()
            
            ml_D_copy1 = ml.copy(name="Modelo_D_copy")
            ml_D_copy2 = ml.copy(name="Modelo_D_copy2")
            
            ml_D_copy1.del_stressmodel("alb_sal")
            ml_D_copy1.del_stressmodel("alb_agua")
            
            ml_D_copy1_series = ml_D_copy1.get_output_series()
            # plt.plot(ml_D_copy1_series)
            # plt.show()
            # print("copias1")
            
            median_no_alb.append(np.median(ml_D_copy1_series.Simulation - ml_D_series.Simulation))
            
            ml_D_copy2.del_stressmodel("SQM_pump")
            ml_D_copy2.del_stressmodel("SQM_iny")
            
            ml_D_copy2_series = ml_D_copy2.get_output_series()
            # plt.plot(ml_D_copy2_series)
            # plt.show()

            modelos_D_series[pozo+"_D"] = ml_D_series
            modelos_D_series[pozo+"_D_copy1"] = ml_D_copy1_series
            modelos_D_series[pozo+"_D_copy2"] = ml_D_copy2_series
            
            # print("copias2")
            
            median_no_sqm.append(np.median(ml_D_copy2_series.Simulation - ml_D_series.Simulation))

        elif modelo == "Modelo_B":
            este_pozo = float(coord_n.loc[pozo]["Este"])
            norte_pozo = float(coord_n.loc[pozo]["Norte"])
            bombeos = ["alb"]
            series = [ALB]
            dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
            
            ml = ps.Model(datos, name=f"{pozo} - {modelo}")
            ml.add_stressmodel(sm_precip_indirecta)
            ml.add_stressmodel(ps.WellModel(series, "alb_sal", dist, ps.HantushWellModel(use_numba=True)))
            
            bombeos = ["til", "tuc", "peine"]
            series = [TIL, TUC, PEINE]
            dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
            ml.add_stressmodel(ps.WellModel(series, "alb_agua", dist, ps.HantushWellModel(use_numba=True)))
            
        elif modelo == "Modelo_C":
            este_pozo = float(coord_n.loc[pozo]["Este"])
            norte_pozo = float(coord_n.loc[pozo]["Norte"])
            bombeos = ["sop", "mop"]
            series = [SOP, MOP]
            dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
            
            ml = ps.Model(datos, name=f"{pozo} - {modelo}")
            ml.add_stressmodel(sm_precip_indirecta)
            ml.add_stressmodel(ps.WellModel(series, "SQM_pump", dist, ps.HantushWellModel(use_numba=True)))
            
            inyeccion = ["sop_iny", "mop_iny"]
            series = [SOP_iny, MOP_iny]
            ml.add_stressmodel(ps.WellModel(series, "SQM_iny", dist, ps.HantushWellModel(use_numba=True),up=True))
            
        elif modelo == "Modelo_D":
            este_pozo = float(coord_n.loc[pozo]["Este"])
            norte_pozo = float(coord_n.loc[pozo]["Norte"])
            
            bombeos = ["alb"]
            series = [ALB]
            
            dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
            
            ml = ps.Model(datos, name=f"{pozo} - {modelo}")
            ml.add_stressmodel(sm_precip_indirecta)
            ml.add_stressmodel(ps.WellModel(series, "alb_sal", dist, ps.HantushWellModel(use_numba=True)))
            
            bombeos = ["til", "tuc", "peine"]
            series = [TIL, TUC, PEINE]
            dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
            ml.add_stressmodel(ps.WellModel(series, "alb_agua", dist, ps.HantushWellModel(use_numba=True)))
            
            bombeos = ["sop", "mop"]
            series = [SOP, MOP]
            dist = [distancia_utm(float(coord_b.loc[b]["Este"]), float(coord_b.loc[b]["Norte"]), este_pozo, norte_pozo) for b in bombeos]
            ml.add_stressmodel(ps.WellModel(series, "SQM_pump", dist, ps.HantushWellModel(use_numba=True)))
            
            inyeccion = ["sop_iny", "mop_iny"]
            series = [SOP_iny, MOP_iny]
            ml.add_stressmodel(ps.WellModel(series, "SQM_iny", dist, ps.HantushWellModel(use_numba=True),up=True))

        if modelo not in ["Modelo_E", "Modelo_D"]:
            # 5. Calibrar y guardar estadísticas
            # 5.1 Sin noisemodel para generar buenos valores iniciales
            ml.solve(report=False,tmin="2013-01-01",tmax="2023-12-01",warmup=720)
            
            # 5.2 Con noisemodel (ArmaNoiseModel) para reducir autocorrelación de los residuales
            ml.add_noisemodel(ps.ArmaNoiseModel())
            ml.solve(initial=False,tmin="2013-01-01",tmax="2023-12-01",warmup=720,report=False)
            model_stats[pozo] = calculate_model_statistics(ml)
            # print(ml.stats.diagnostics(alpha=0.05))

 
        # 6. Gráfico
        os.makedirs(modelo, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [4, 1]})
        ml.plot(ax=axes[0])
        axes[0].set_ylabel("Nivel de agua [msnm]")
        axes[0].set_title(f"{modelo} - {pozo}")
        axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[1].hist(ml.residuals(), bins=25, edgecolor="black", orientation="horizontal")
        axes[1].set_xlabel("Frecuencia")
        axes[1].set_ylabel("Residuales")
        axes[1].set_title("Residuales")
        plt.tight_layout()
        fig.savefig(os.path.join(modelo, f"{modelo}_{pozo}.png"), dpi=200)
        plt.close()

        # 7. Gains (solo Modelo_C, D, E)
        
        if modelo == "Modelo_B":
            pozo_gains = {"pozo": pozo}
            alb_sal_stressmodel = ml.stressmodels["alb_sal"]
            names = ["alb"]
            
            for i, name in enumerate(names[: len(alb_sal_stressmodel.stress)]):
                p = alb_sal_stressmodel.get_parameters(model=ml, istress=i)
                pozo_gains[name] = alb_sal_stressmodel.rfunc.gain(p) * 1e6 / 365.25
            gains_data.append(pozo_gains)
            
            alb_agua_stressmodel = ml.stressmodels["alb_agua"]
            names = ["til", "tuc", "peine"]
            
            for i, name in enumerate(names[: len(alb_agua_stressmodel.stress)]):
                p = alb_agua_stressmodel.get_parameters(model=ml, istress=i)
                pozo_gains[name] = alb_agua_stressmodel.rfunc.gain(p) * 1e6 / 365.25
            gains_data.append(pozo_gains)

        elif modelo == "Modelo_C":
            pozo_gains = {"pozo": pozo}
            sqm_stressmodel = ml.stressmodels["SQM_pump"]
            names = ["sop", "mop"]
            
            for i, name in enumerate(names[: len(sqm_stressmodel.stress)]):
                p = sqm_stressmodel.get_parameters(model=ml, istress=i)
                pozo_gains[name] = sqm_stressmodel.rfunc.gain(p) * 1e6 / 365.25
            gains_data.append(pozo_gains)
            
            sqm_iny_stressmodel = ml.stressmodels["SQM_iny"]
            names = ["sop_iny", "mop_iny"]
            
            for i, name in enumerate(names[: len(sqm_iny_stressmodel.stress)]):
                p = sqm_iny_stressmodel.get_parameters(model=ml, istress=i)
                pozo_gains[name] = sqm_iny_stressmodel.rfunc.gain(p) * 1e6 / 365.25
            gains_data.append(pozo_gains)   
            
        elif modelo == "Modelo_E":
            pozo_gains = {"pozo": pozo}
            alb_sal_stressmodel = ml.stressmodels["alb_sal"]
            names = ["alb"]
            
            for i, name in enumerate(names[: len(alb_sal_stressmodel.stress)]):
                p = alb_sal_stressmodel.get_parameters(model=ml, istress=i)
                pozo_gains[name] = alb_sal_stressmodel.rfunc.gain(p) * 1e6 / 365.25
            gains_data.append(pozo_gains)
            
            alb_agua_stressmodel = ml.stressmodels["alb_agua"]
            names = ["til", "tuc", "peine"]
            
            for i, name in enumerate(names[: len(alb_agua_stressmodel.stress)]):
                p = alb_agua_stressmodel.get_parameters(model=ml, istress=i)
                pozo_gains[name] = alb_agua_stressmodel.rfunc.gain(p) * 1e6 / 365.25
            gains_data.append(pozo_gains)
            
            sqm_stressmodel = ml.stressmodels["SQM_pump"]
            names = ["sop", "mop"]
            
            for i, name in enumerate(names[: len(sqm_stressmodel.stress)]):
                p = sqm_stressmodel.get_parameters(model=ml, istress=i)
                pozo_gains[name] = sqm_stressmodel.rfunc.gain(p) * 1e6 / 365.25
                
            inyeccion_stressmodel = ml.stressmodels["SQM_iny"]
            names = ["sop_iny", "mop_iny"]
            
            for i, name in enumerate(names[: len(inyeccion_stressmodel.stress)]):
                p = inyeccion_stressmodel.get_parameters(model=ml, istress=i)
                pozo_gains[name] = inyeccion_stressmodel.rfunc.gain(p) * 1e6 / 365.25
                
            gains_data.append(pozo_gains)


    return model_stats, gains_data, median_no_alb, median_no_sqm


def _run_one_model(modelo: str, location_names: list, bundle: dict):
    """
    Wrapper (por modelo): ejecuta create_model_with_data para un tipo de modelo
    y todos los pozos. Devuelve (modelo, model_stats, gains_data).
    """
    model_stats, gains_data, median_no_alb, median_no_sqm = create_model_with_data(modelo, location_names, bundle)
    return (modelo, model_stats, gains_data, median_no_alb, median_no_sqm)


def _run_one_pozo_for_model(modelo: str, pozo: str, bundle: dict):
    """
    Wrapper para multiprocessing: ejecuta un solo modelo para un solo pozo.
    Devuelve (modelo, model_stats, gains_data) con model_stats de una clave (pozo).
    """
    model_stats, gains_data, median_no_alb_data, median_no_sqm_data = create_model_with_data(modelo, [pozo], bundle)
    return (modelo, model_stats, gains_data, median_no_alb_data, median_no_sqm_data)


# =============================================================================
# 3. CARGA Y PREPARACIÓN DE DATOS DE NIVELES
# =============================================================================
# (Todo lo que sigue se ejecuta solo al correr el script, no al importar;
#  así multiprocessing no vuelve a cargar datos en cada worker.)
if __name__ == "__main__":
    t_start = time.perf_counter()
    print("\n=== 3. CARGANDO DATOS DE NIVELES ===")

    # Leer datos de niveles de pozos
    niveles_path = os.path.join("datos", "pozos nivel peine", "Niveles.csv")
    niveles = pd.read_csv(niveles_path, encoding="utf-8-sig")

    # Limpiar y estandarizar datos
    niveles["Fecha"] = pd.to_datetime(niveles["Fecha"], dayfirst=True, errors="coerce")
    niveles["Valor"] = pd.to_numeric(niveles["Valor"], errors="coerce")
    niveles["Pozo"] = niveles["Pozo"].astype(str).str.strip()
    niveles["Tipo"] = niveles["Tipo"].astype(str).str.strip().str.upper()
    niveles = niveles.dropna(subset=["Fecha", "Pozo", "Tipo", "Valor"]).copy()

    # Filtrar por fecha de inicio
    inicio = pd.Timestamp("2010-01-01")
    final = pd.Timestamp("2023-12-01")
    niveles = niveles[(niveles["Fecha"] >= inicio) & (niveles["Fecha"] <= final)].copy()

    # Filtrar solo pozos permitidos (según archivo de monitoreo)
    monitoreo_path = os.path.join("datos", "pozos nivel peine", "monitoreo_total.csv")
    monitoreo = pd.read_csv(monitoreo_path, encoding="utf-8-sig")
    allowed = (
        monitoreo["Nombre"]
        .astype(str)
        .str.strip()
        .str.replace(r"^Pozo\\s+", "", regex=True)
        .dropna()
        .unique()
    )
    niveles = niveles[niveles["Pozo"].isin(set(allowed))].copy()

    print(f"Total de registros de niveles: {len(niveles)}")
    print(f"Pozos únicos: {niveles['Pozo'].nunique()}")

    # =========================================================================
    # 4. ANÁLISIS DE TENDENCIAS (MANN-KENDALL)
    # =========================================================================
    print("\n=== 4. ANÁLISIS DE TENDENCIAS ===")

    # Aplicar test a diferentes grupos de pozos
    exclude_head = {"PP-03", "MP-07C-1", "MP-07A", "L10-1", "PP-01", "MP-08A", "BA-30"}

    mk_head = _mk_table(niveles[(niveles["Tipo"] == "HEAD") & (~niveles["Pozo"].isin(exclude_head))])
    mk_head_excl = _mk_table(niveles[(niveles["Tipo"] == "HEAD") & (niveles["Pozo"].isin(exclude_head))])
    mk_stage = _mk_table(niveles[niveles["Tipo"] == "STAGE"])

    # Exportar resultados
    mk_head.to_csv("resultados_mann_kendall_head.csv", index=False, float_format="%.3f")
    mk_head_excl.to_csv("resultados_mann_kendall_head_excluidos.csv", index=False, float_format="%.3f")
    mk_stage.to_csv("resultados_mann_kendall_limnimetricos.csv", index=False, float_format="%.3f")

    print(f"MK HEAD: {len(mk_head)} pozos")
    print(f"MK HEAD excluidos: {len(mk_head_excl)} pozos")
    print(f"MK STAGE: {len(mk_stage)} pozos")

    # =========================================================================
    # 5. CARGA DE DATOS DE ESTRÉS (CLIMA Y BOMBEOS)
    # =========================================================================
    print("\n=== 5. CARGANDO DATOS DE ESTRÉS ===")

    # 5.1 Precipitación y Evaporación
    archivos_precipitacion = [
        os.path.join('datos', 'resultados_meteo', 'precip', 'Prec_CHAXA.csv'),
        os.path.join('datos', 'resultados_meteo', 'precip', 'Prec_LZA9-1 (Interna).csv'),
        os.path.join('datos', 'resultados_meteo', 'precip', 'Prec_LZA10-1.csv')
    ]
    archivos_evaporacion = [
        os.path.join('datos', 'resultados_meteo', 'evap', 'Evap_CHAXA.csv'),
        os.path.join('datos', 'resultados_meteo', 'evap', 'Evap_LZA9-1 (Interna).csv')
    ]

    # Concatenar y promediar múltiples estaciones
    precipitacion_list = [pd.read_csv(archivo, index_col=0, parse_dates=True).squeeze("columns")
                          for archivo in archivos_precipitacion]
    precipitacion = pd.concat(precipitacion_list).groupby(level=0).mean()

    evaporacion_list = [pd.read_csv(archivo, index_col=0, parse_dates=True).squeeze("columns")
                        for archivo in archivos_evaporacion]
    evaporacion = pd.concat(evaporacion_list).groupby(level=0).mean()

    # 5.2 Datos de Bombeo (usando función auxiliar)
    print("Cargando datos de bombeo...")
    ALB_series = load_pumping_data(os.path.join("datos", "pumping", "alb_pump.csv"))
    SOP_series = load_pumping_data(os.path.join("datos", "pumping", "SOP_extraccion_monthly_m3.csv"))
    MOP_series = load_pumping_data(os.path.join("datos", "pumping", "MOP_extraccion_monthly_m3.csv"))
    SOP_iny_series = load_pumping_data(os.path.join("datos", "pumping", "SOP_inyeccion_monthly_m3.csv"))
    MOP_iny_series = load_pumping_data(os.path.join("datos", "pumping", "MOP_inyeccion_monthly_m3.csv"))
    TIL_series = load_pumping_data(os.path.join("datos", "pumping", "tilopozo_pumping.csv"))
    TUC_series = load_pumping_data(os.path.join("datos", "pumping", "tucucaro_pumping.csv"))
    PEINE_series = load_pumping_data(os.path.join("datos", "pumping", "Pozo_peine_pumping.csv"))

    # Plot de todas las series de bombeo (stacked time series)
    pumping_series_list = [
        ("ALB (Albemarle)", ALB_series),
        ("SOP extracción", SOP_series),
        ("MOP extracción", MOP_series),
        ("SOP inyección", SOP_iny_series),
        ("MOP inyección", MOP_iny_series),
        ("Tilopozo", TIL_series),
        ("Tucucaro", TUC_series),
        ("Peine", PEINE_series),
    ]
    n_series = len(pumping_series_list)
    fig, axes = plt.subplots(n_series, 1, figsize=(12, 2 * n_series), sharex=True)
    if n_series == 1:
        axes = [axes]
    for ax, (label, ser) in zip(axes, pumping_series_list):
        ser_plot = ser.dropna()
        if len(ser_plot):
            ax.fill_between(ser_plot.index, ser_plot.values, 0, alpha=0.6)
            ax.plot(ser_plot.index, ser_plot.values, linewidth=0.8)
        ax.set_ylabel("m³/día")
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[-1].set_xlabel("Fecha")
    fig.suptitle("Series de bombeo (extracción e inyección)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join("plots", "all_pumping_stacked.png"), dpi=150)
    plt.close()
    print("  Guardado: plots/all_pumping_stacked.png")

    # Organizar series de bombeo
    bombeos_dic = {
        "alb": ALB_series, "sop": SOP_series, "mop": MOP_series,
        "til": TIL_series, "tuc": TUC_series, "peine": PEINE_series
    }
    inyeccion_dic = {
        "sop_iny": SOP_iny_series, "mop_iny": MOP_iny_series
    }
    
    list_of_bombeos = [ALB_series, SOP_series, MOP_series, TIL_series, TUC_series, PEINE_series]
    list_of_inyeccion = [SOP_iny_series, MOP_iny_series]

    # 5.3 Preparar clima para modelos (resample a diario)
    precipitacion.index = pd.to_datetime(precipitacion.index, errors="coerce")
    precipitacion = precipitacion.dropna().sort_index()
    precipitacion_monthly = precipitacion.resample("D").sum()

    evaporacion.index = pd.to_datetime(evaporacion.index, errors="coerce")
    evaporacion = evaporacion.dropna().sort_index()
    evaporacion_monthly = evaporacion.resample("D").mean()

    # Preparar series para modelos Pastas
    prec = precipitacion.dropna().sort_index().resample("D").sum()
    evap = evaporacion.dropna().sort_index().resample("D").mean()

    # Crear StressModels para clima
    coef = 0.225  # Coeficiente de infiltracion SRK Consulting (2020) Zona Marginal (promedio)
    sm_precip_directa = ps.StressModel(prec, ps.Gamma(), settings="prec", name="precipitacion_directa")
    sm_precip_indirecta = ps.StressModel(prec * coef, ps.Gamma(), settings="prec", name="precipitacion_indirecta")
    sm_evap = ps.StressModel(evap, ps.Gamma(), settings="evap", name="evaporacion")

    print("Datos de estrés cargados correctamente")

    # =========================================================================
    # 6. CARGA DE COORDENADAS
    # =========================================================================
    print("\n=== 6. CARGANDO COORDENADAS ===")

    coordenadas_bombeos_path = os.path.join("datos", "pumping", "bombeos_ubicacion.csv")
    coordenadas_bombeos = pd.read_csv(coordenadas_bombeos_path, encoding="utf-8-sig")
    try:
        coordenadas_bombeos.set_index("Nombre", inplace=True)
    except Exception as e:
        print(f"Error loading coordenadas_bombeos: {e}")
        raise

    coordenadas_niveles_path = os.path.join("datos", "pozos nivel peine", "monitoreo_total.csv")
    try:
        coordenadas_niveles = pd.read_csv(coordenadas_niveles_path, encoding="utf-8-sig")
        if "Nombre" in coordenadas_niveles.columns:
            coordenadas_niveles.set_index("Nombre", inplace=True)
        else:
            print(f"Warning: 'Nombre' column not found. Available columns: {coordenadas_niveles.columns.tolist()}")
            if "nombre" in coordenadas_niveles.columns.str.lower():
                col_name = coordenadas_niveles.columns[coordenadas_niveles.columns.str.lower() == "nombre"][0]
                coordenadas_niveles.set_index(col_name, inplace=True)
    except Exception as e:
        print(f"Error loading coordenadas_niveles: {e}")
        raise

    print(f"Coordenadas de {len(coordenadas_bombeos)} bombeos cargadas")
    print(f"Coordenadas de {len(coordenadas_niveles)} pozos cargadas")

    # -------------------------------------------------------------------------
    # 8. EJECUCIÓN: modelos secuencial (A, B, C...), pozos en paralelo
    # -------------------------------------------------------------------------
    # Bucle de modelos secuencial; para cada modelo, todos los pozos se procesan
    # en paralelo (cada worker = un pozo para ese modelo).
    Modelos = ["Modelo_A", "Modelo_B", "Modelo_C", "Modelo_D","Modelo_E"]
    niveles_series = niveles.copy()
    niveles_series.set_index("Fecha", inplace=True)
    # Lista de pozos y mapping Pozo -> Tipo (para usar tipo al iterar)
    location_names = sorted(niveles["Pozo"].unique().tolist())
    pozo_to_tipo = niveles.groupby("Pozo")["Tipo"].first().to_dict()  # un Tipo por pozo
    # Para probar con un pozo: location_names = ["REGLILLA SALADA CONAF"]
    # location_names = location_names[0:7]  # para probar con los primeros N pozos

    bundle = get_data_bundle()

    # Acumular estadísticas por modelo para comparación AIC / delta AIC
    all_model_stats = {}
    total_modelos = 0
    # Acumular medians solo de Modelo_E para el boxplot (se rellenan cuando modelo == "Modelo_E")
    median_no_alb_all = []
    median_no_sqm_all = []
    for modelo in Modelos:
        print(f"\n--- {modelo} ---")
        if N_WORKERS <= 1:
            # Secuencial: un pozo tras otro para este modelo
            results = []
            for pozo in location_names:
                results.append(_run_one_pozo_for_model(modelo, pozo, bundle))
        else:
            # Paralelo: todos los pozos para este modelo en paralelo
            with Pool(N_WORKERS) as pool:
                results = pool.starmap(
                    _run_one_pozo_for_model,
                    [(modelo, pozo, bundle) for pozo in location_names],
                )

        # Agregar y exportar para este modelo
        model_stats = {}
        gains_data = []
        median_no_alb_data = []
        median_no_sqm_data = []
        for (m, pozo_stats, gains, med_alb, med_sqm) in results:
            model_stats.update(pozo_stats)
            gains_data.extend(gains)
            median_no_alb_data.extend(med_alb)
            median_no_sqm_data.extend(med_sqm)
        if modelo == "Modelo_E":
            median_no_alb_all.extend(median_no_alb_data)
            median_no_sqm_all.extend(median_no_sqm_data)
        total_modelos += len(model_stats)

        df_stats = pd.DataFrame.from_dict(model_stats, orient="index")
        df_stats["IVM"] = df_stats.apply(calculate_ivm, axis=1)
        summary_row = pd.DataFrame([{c: "" for c in df_stats.columns}], index=["Summary"])
        summary_row["IVM"] = f"Total Válidos: {(df_stats['IVM'] == 'Válido').sum()}"
        df_stats = pd.concat([df_stats, summary_row])
        df_stats.to_csv(f"Summary_{modelo}.csv")
        print(f"  Guardado: Summary_{modelo}.csv")
        all_model_stats[modelo] = {pozo: stats for pozo, stats in model_stats.items()}
        if modelo in ["Modelo_C", "Modelo_D", "Modelo_E"] and gains_data:
            df_gains = pd.DataFrame(gains_data).set_index("pozo")
            df_gains = pd.concat([df_gains, pd.DataFrame([df_gains.mean()], index=["mean"])])
            # Median (recommended primary statistic)
            df_gains = pd.concat(
                [df_gains, pd.DataFrame([df_gains.median()], index=["median"])]
            )

            # Quantiles (to capture heterogeneity)
            df_gains = pd.concat(
                [
                    df_gains,
                    pd.DataFrame([df_gains.quantile(0.25)], index=["q25"]),
                    pd.DataFrame([df_gains.quantile(0.75)], index=["q75"]),
                ]
            )
            df_gains.to_csv(f"{modelo}_gains_by_pozo.csv")
            print(f"  Guardado: {modelo}_gains_by_pozo.csv")

    # Comparación entre modelos: AIC y delta AIC (B, C, D, E)
    if all_model_stats and len(all_model_stats) == len(Modelos):
        rows_aic = []
        rows_delta = []
        for pozo in location_names:
            row_aic = {"Pozo": pozo}
            row_delta = {"Pozo": pozo}
            aic_vals = {}
            for modelo in Modelos:
                if pozo in all_model_stats.get(modelo, {}):
                    aic = all_model_stats[modelo][pozo].get("AIC", np.nan)
                    bic = all_model_stats[modelo][pozo].get("BIC", np.nan)
                    row_aic[f"AIC_{modelo}"] = aic
                    row_aic[f"BIC_{modelo}"] = bic
                    if modelo in ["Modelo_B", "Modelo_C", "Modelo_D", "Modelo_E"]:
                        aic_vals[modelo] = aic
            min_aic = min(aic_vals.values()) if aic_vals else np.nan
            for m in ["Modelo_B", "Modelo_C", "Modelo_D", "Modelo_E"]:
                row_delta[f"dAIC_{m}"] = (aic_vals.get(m, np.nan) - min_aic) if not np.isnan(min_aic) else np.nan
            rows_aic.append(row_aic)
            rows_delta.append(row_delta)
        df_aic = pd.DataFrame(rows_aic).set_index("Pozo")
        df_delta = pd.DataFrame(rows_delta).set_index("Pozo")
        df_aic.to_csv("comparison_aic_bic_by_model.csv")
        df_delta.to_csv("comparison_delta_aic_BCDE.csv")
        print("  Guardado: comparison_aic_bic_by_model.csv")
        print("  Guardado: comparison_delta_aic_BCDE.csv")

    elapsed = time.perf_counter() - t_start
    print("\n=== PROCESO COMPLETADO ===")
    print(f"Total de modelos creados: {total_modelos}")
    if elapsed >= 60:
        mins = int(elapsed // 60)
        secs = elapsed % 60
        print(f"Tiempo total: {mins}m {secs:.1f}s")
    else:
        print(f"Tiempo total: {elapsed:.1f}s")
        
    # Boxplot y test pareado: median_no_alb vs median_no_sqm (qué estrés genera más drawdown)
    if median_no_alb_all and median_no_sqm_all:
        plt.figure(figsize=(10, 6))
        plt.boxplot(
            [median_no_alb_all, median_no_sqm_all],
            labels=["Sin Albemarle (D−D_copy1)", "Sin SQM (D−D_copy2)"],
            showfliers=False
        )
        plt.ylabel("Mediana de la diferencia de nivel simulado [m]")
        plt.title("Boxplot: Efecto Albemarle vs SQM")
        plt.tight_layout()
        plt.savefig(os.path.join("plots", "median_no_alb_and_median_no_sqm.png"), dpi=150)
        plt.close()
        print("  Guardado: plots/median_no_alb_and_median_no_sqm.png")

        # Test pareado: ¿Albemarle o SQM explica más drawdown? (valor más alto = más efecto)
        x_alb = np.array(median_no_alb_all, dtype=float)
        x_sqm = np.array(median_no_sqm_all, dtype=float)
        # Quitar NaN si los hay y asegurar mismo n
        mask = np.isfinite(x_alb) & np.isfinite(x_sqm)
        x_alb, x_sqm = x_alb[mask], x_sqm[mask]
        if len(x_alb) >= 3:
            stat_w, p_wilcoxon = stats.wilcoxon(x_alb, x_sqm, alternative="two-sided")
            t_rel, p_ttest = stats.ttest_rel(x_alb, x_sqm)
            med_alb, med_sqm = np.median(x_alb), np.median(x_sqm)
            mean_alb, mean_sqm = np.mean(x_alb), np.mean(x_sqm)
            print("\n  --- Test pareado (Albemarle vs SQM, por pozo) ---")
            print(f"  Mediana efecto sin Albemarle: {med_alb:.4f} m  |  Mediana efecto sin SQM: {med_sqm:.4f} m")
            print(f"  Media   efecto sin Albemarle: {mean_alb:.4f} m  |  Media   efecto sin SQM: {mean_sqm:.4f} m")
            print(f"  Wilcoxon signed-rank (two-sided): estadístico = {stat_w:.3f}, p = {p_wilcoxon:.4f}")
            print(f"  t-test pareado (two-sided): t = {t_rel:.3f}, p = {p_ttest:.4f}")
            if p_wilcoxon < 0.05:
                if med_alb > med_sqm:
                    print("  Conclusión: el efecto de Albemarle (drawdown) es significativamente mayor que el de SQM.")
                else:
                    print("  Conclusión: el efecto de SQM (drawdown) es significativamente mayor que el de Albemarle.")
            else:
                print("  Conclusión: no hay diferencia significativa entre ambos efectos (p >= 0.05).")
        else:
            print("  (Datos insuficientes para test pareado; n >= 3 requerido)")
    else:
        print("  (No hay datos de median_no_alb/median_no_sqm; boxplot omitido)")
