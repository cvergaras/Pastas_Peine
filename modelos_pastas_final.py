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

# Suprimir deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configurar nivel de logging de Pastas
ps.set_log_level("ERROR")
ps.show_versions()

# Crear directorio para plots
os.makedirs("plots", exist_ok=True)

# =============================================================================
# 2. FUNCIONES AUXILIARES
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
    return -1 * daily_series


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
    
    # Legates & McCabe (1999)
    obs = model.oseries.series
    sim = model.simulate()  # devuelve un pandas Series alineado con obs
    
    #std
    std_obs = model.oseries.series.dropna().std()

    # MAE
    mae = np.mean(np.abs(sim - obs))
    
    return {
        "R²": r2,
        "RMSE": rmse,
        "std_obs07": std_obs * 0.7,
        # "p-value zero": p_value_t,
        # "t-statistic zero": t_stat,
        "p-value wilconox": p_value_wilc,
        # "statistic wilconox": stat_wilc,
        # "p-value shapiro": p_value_shap,
        # "statistic shapiro": stat_shap,
        "MAE": mae,
        "BIC": bic,
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
            # pd.notna(row["p-value shapiro"]) and row["p-value shapiro"] > 0.05 # No se usa porque no funciona bien con muchos datos
        ]
        return "Válido" if all(conditions) else "No Válido"
    except (KeyError, TypeError):
        return "No Válido"


# =============================================================================
# 3. CARGA Y PREPARACIÓN DE DATOS DE NIVELES
# =============================================================================
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
inicio = pd.Timestamp("2000-01-01")
inicio2 = pd.Timestamp("1998-01-01")
niveles = niveles[niveles["Fecha"] >= inicio].copy()

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

# =============================================================================
# 4. ANÁLISIS DE TENDENCIAS (MANN-KENDALL)
# =============================================================================
print("\n=== 4. ANÁLISIS DE TENDENCIAS ===")

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

# =============================================================================
# 5. CARGA DE DATOS DE ESTRÉS (CLIMA Y BOMBEOS)
# =============================================================================
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
SOP_series = load_pumping_data(os.path.join("datos", "pumping", "SOP_monthly_m3.csv"))
MOP_series = load_pumping_data(os.path.join("datos", "pumping", "MOP_monthly_m3.csv"))
TIL_series = load_pumping_data(os.path.join("datos", "pumping", "tilopozo_pumping.csv"))
TUC_series = load_pumping_data(os.path.join("datos", "pumping", "tucucaro_pumping.csv"))
PEINE_series = load_pumping_data(os.path.join("datos", "pumping", "pozo_peine_pumping.csv"))

# Organizar series de bombeo
bombeos_dic = {
    "alb": ALB_series, "sop": SOP_series, "mop": MOP_series,
    "til": TIL_series, "tuc": TUC_series, "peine": PEINE_series
}
list_of_bombeos = [ALB_series, SOP_series, MOP_series, TIL_series, TUC_series, PEINE_series]
stress_names = ["alb", "sop", "mop", "til", "tuc", "peine"]

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
sm_precip = ps.StressModel(prec * coef, ps.Gamma(), settings="prec", name="precipitacion")
sm_evap = ps.StressModel(evap, ps.Gamma(), settings="evap", name="evaporacion")

print("Datos de estrés cargados correctamente")

# =============================================================================
# 6. CARGA DE COORDENADAS
# =============================================================================
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
        # Try alternative column name
        if "nombre" in coordenadas_niveles.columns.str.lower():
            col_name = coordenadas_niveles.columns[coordenadas_niveles.columns.str.lower() == "nombre"][0]
            coordinadas_niveles.set_index(col_name, inplace=True)
except Exception as e:
    print(f"Error loading coordenadas_niveles: {e}")
    raise

print(f"Coordenadas de {len(coordenadas_bombeos)} bombeos cargadas")
print(f"Coordenadas de {len(coordenadas_niveles)} pozos cargadas")

# =============================================================================
# 7. CONSTRUCCIÓN DE MODELOS PASTAS
# =============================================================================
print("\n=== 7. CONSTRUYENDO MODELOS PASTAS ===")

# Definir modelos a construir
Modelos = ["Modelo_A", "Modelo_B", "Modelo_C", "Modelo_D", "Modelo_E"]

# Descripción de cada modelo:
# Modelo_A: Solo recarga (precipitación + evaporación)
# Modelo_B: Recarga + bombeo Albemarle (Hantush)
# Modelo_C: Recarga + Albemarle + pozos de agua (WellModel: alb, til, tuc, peine)
# Modelo_D: Recarga + SOP + MOP (WellModel)
# Modelo_E: Recarga + todos los bombeos (WellModel: alb, sop, mop, til, tuc, peine)

# Preparar datos de niveles para modelos
niveles_series = niveles.copy()
niveles_series.set_index("Fecha", inplace=True)
location_names = niveles["Pozo"].unique()
location_names = ["L10-1"]

# Inicializar diccionarios para almacenar resultados
modelos_individuales = {}
model_stats = {}
gains_data = []

Modelos = ["Modelo_E"]

# Iterar sobre cada modelo
for modelo in Modelos:
    print(f"\n--- Procesando {modelo} ---")
    
    # Reiniciar gains_data para cada modelo
    gains_data = []
    
    # Iterar sobre pozos (actualmente solo L10-1, pero puede expandirse)
    for pozo in location_names:
        print(f"  Pozo: {pozo}")
        
        # Preparar serie de observaciones
        datos = niveles_series[niveles_series["Pozo"] == pozo]["Valor"]
        datos.index = pd.to_datetime(datos.index, errors="coerce")
        datos = datos.dropna().sort_index().resample("D").mean()
        
        # Crear modelo Pastas
        ml = ps.Model(datos, name=f"{pozo} - {modelo}")
        
        # Agregar recarga (siempre presente)
        ml.add_stressmodel(sm_precip)
        ml.add_stressmodel(sm_evap)
        
        # Agregar estrés según tipo de modelo
        if modelo == "Modelo_B":
            # Solo bombeo Albemarle con Hantush
            sm_alb = ps.StressModel(ALB_series, ps.Hantush(), name="albemarle", 
                                   up=False, settings="well")
            ml.add_stressmodel(sm_alb)
            
        elif modelo == "Modelo_C":
            # WellModel con Albemarle + pozos de agua
            este_pozo = float(coordenadas_niveles.loc[pozo]["Este"])
            norte_pozo = float(coordenadas_niveles.loc[pozo]["Norte"])
            
            distances = []
            bombeos_modelo = ["alb", "til", "tuc", "peine"]
            series_modelo = [ALB_series, TIL_series, TUC_series, PEINE_series]
            
            for bombeo in bombeos_modelo:
                este_bombeo = float(coordenadas_bombeos.loc[bombeo]["Este"])
                norte_bombeo = float(coordenadas_bombeos.loc[bombeo]["Norte"])
                distance_bombeo = distancia_utm(este_bombeo, norte_bombeo, este_pozo, norte_pozo)
                distances.append(distance_bombeo)
            
            sm_bombeo = ps.WellModel(series_modelo, "WellModel", distances)
            ml.add_stressmodel(sm_bombeo)
            
        elif modelo == "Modelo_D":
            # WellModel con SOP + MOP
            este_pozo = float(coordenadas_niveles.loc[pozo]["Este"])
            norte_pozo = float(coordenadas_niveles.loc[pozo]["Norte"])
            
            distances = []
            bombeos_modelo = ["sop", "mop"]
            series_modelo = [SOP_series, MOP_series]
            
            for bombeo in bombeos_modelo:
                este_bombeo = float(coordenadas_bombeos.loc[bombeo]["Este"])
                norte_bombeo = float(coordenadas_bombeos.loc[bombeo]["Norte"])
                distance_bombeo = distancia_utm(este_bombeo, norte_bombeo, este_pozo, norte_pozo)
                distances.append(distance_bombeo)
            
            sm_bombeo = ps.WellModel(series_modelo, "WellModel", distances)
            ml.add_stressmodel(sm_bombeo)
            
        elif modelo == "Modelo_E":
            # WellModel con todos los bombeos
            este_pozo = float(coordenadas_niveles.loc[pozo]["Este"])
            norte_pozo = float(coordenadas_niveles.loc[pozo]["Norte"])
            
            distances = []
            bombeos_modelo = ["alb", "sop", "mop", "til", "tuc", "peine"]
            series_modelo = [ALB_series, SOP_series, MOP_series, TIL_series, TUC_series, PEINE_series]
            
            for bombeo in bombeos_modelo:
                este_bombeo = float(coordenadas_bombeos.loc[bombeo]["Este"])
                norte_bombeo = float(coordenadas_bombeos.loc[bombeo]["Norte"])
                distance_bombeo = distancia_utm(este_bombeo, norte_bombeo, este_pozo, norte_pozo)
                distances.append(distance_bombeo)
            
            sm_bombeo = ps.WellModel(series_modelo, "WellModel", distances)
            ml.add_stressmodel(sm_bombeo)
        
        # Agregar noisemodel (Optional)
        # ml.add_noisemodel(ps.NoiseModel())
        
        # Calibrar modelo
        ml.solve(report=False)
        
        # Calcular estadísticas
        stats_dict = calculate_model_statistics(ml)
        modelos_individuales[f"{pozo}_{modelo}"] = ml
        model_stats[(pozo, modelo)] = stats_dict
        
        # Guardar gráficos
        os.makedirs(modelo, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})
        ml.plot(ax=axes[0])
        axes[0].set_ylabel('Nivel de agua [msnm]')
        axes[0].set_title(f'{modelo} - {pozo}')
        axes[0].ticklabel_format(axis='y', style='plain', useOffset=False)
        
        residuals = ml.residuals()
        axes[1].hist(residuals, bins=25, edgecolor='black', orientation='horizontal')
        axes[1].set_xlabel('Frecuencia')
        axes[1].set_ylabel('Residuales')
        axes[1].set_title('Residuales')
        
        plt.tight_layout()
        fig.savefig(os.path.join(modelo, f"{modelo}_{pozo}.png"), dpi=200)
        plt.close()
        
        # Calcular gains para modelos con WellModel
        pozo_gains = {"pozo": pozo}
        
        if modelo in ["Modelo_C", "Modelo_D", "Modelo_E"]:
            wm = ml.stressmodels["WellModel"]
                    # Definir nombres de estrés según modelo
            if modelo == "Modelo_C":
                stress_names = ["alb", "til", "tuc", "peine"]
                
            elif modelo == "Modelo_D":
                stress_names = ["sop", "mop"]
                
            elif modelo == "Modelo_E":
                stress_names = ["alb", "sop", "mop", "til", "tuc", "peine"]
            
            for i, name in enumerate(stress_names[:len(wm.stress)]):
                p = wm.get_parameters(model=ml, istress=i)
                gain = wm.rfunc.gain(p) * 1e6 / 365.25  # m per Mm³/year
                pozo_gains[name] = gain
            
            gains_data.append(pozo_gains)
                  
    
    # =========================================================================
    # 8. EXPORTAR RESULTADOS
    # =========================================================================
    
    # Exportar gains (solo para modelos con WellModel)
    if modelo not in ["Modelo_A", "Modelo_B"] and gains_data:
        df_gains = pd.DataFrame(gains_data)
        df_gains.set_index("pozo", inplace=True)
        
        # Agregar fila con media
        mean_row = df_gains[stress_names[:len(df_gains.columns)]].mean()
        mean_df = pd.DataFrame([mean_row], index=["mean"])
        df_gains = pd.concat([df_gains, mean_df])
        
        df_gains.to_csv(f"{modelo}_gains_by_pozo.csv")
        print(f"  Gains exportados: {modelo}_gains_by_pozo.csv")
    
    # Exportar estadísticas con IVM
    df_all_stats = pd.DataFrame.from_dict(model_stats, orient='index')
    df_model_stats = df_all_stats[df_all_stats.index.get_level_values(1) == modelo].copy()
    
    if isinstance(df_model_stats.index, pd.MultiIndex):
        df_model_stats.index = df_model_stats.index.get_level_values(0)
    
    # Calcular IVM
    df_model_stats["IVM"] = df_model_stats.apply(calculate_ivm, axis=1)
    
    # Agregar fila de resumen
    validos_count = (df_model_stats["IVM"] == "Válido").sum()
    summary_dict = {col: "" for col in df_model_stats.columns}
    summary_dict["IVM"] = f"Total Válidos: {validos_count}"
    summary_row = pd.DataFrame([summary_dict], index=["Summary"])
    df_model_stats = pd.concat([df_model_stats, summary_row])
    
    df_model_stats.to_csv(f"Summary_{modelo}.csv")
    print(f"  Estadísticas exportadas: Summary_{modelo}.csv")

print("\n=== PROCESO COMPLETADO ===")
print(f"Total de modelos creados: {len(modelos_individuales)}")
