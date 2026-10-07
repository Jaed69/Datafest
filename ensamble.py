import json
import pandas as pd

def hacer_ensamble():
    print("Buscando las mejores predicciones en index.json...")
    with open('experiments/index.json', 'r') as f:
        d = json.load(f)
    
    # Filtrar corridas por modelo
    lgb_runs = [r for r in d['runs'] if 'lightgbm' in r['model_name']]
    cat_runs = [r for r in d['runs'] if 'catboost' in r['model_name']]
    
    if not lgb_runs or not cat_runs:
        print("Error: Faltan datos de LightGBM o CatBoost en el historial.")
        return

    # Ordenar por puntaje Gini y seleccionar el mejor de cada uno
    mejor_lgb = sorted(lgb_runs, key=lambda x: x['inner_mean_gini'], reverse=True)[0]
    mejor_cat = sorted(cat_runs, key=lambda x: x['inner_mean_gini'], reverse=True)[0]
    
    print(f"Mejor LightGBM encontrado con Gini: {mejor_lgb['inner_mean_gini']:.4f}")
    print(f"Mejor CatBoost encontrado con Gini: {mejor_cat['inner_mean_gini']:.4f}")
    
    # Leer los archivos de predicción
    df_lgb = pd.read_csv(mejor_lgb['submission'])
    df_cat = pd.read_csv(mejor_cat['submission'])
    
    # Crear el ensamble promediando las probabilidades
    # Se detecta dinámicamente el nombre de la columna (ej. 'probabilidad' u 'objetivo')
    col_proba = df_lgb.columns[1] 
    
    df_ensamble = df_lgb.copy()
    df_ensamble[col_proba] = (df_lgb[col_proba] * 0.6) + (df_cat[col_proba] * 0.4)
    
    df_ensamble.to_csv('ENTREGABLE_ENSAMBLE.csv', index=False)
    print("\n¡Éxito! El ensamble se guardó en la raíz como: ENTREGABLE_ENSAMBLE.csv")

if __name__ == "__main__":
    hacer_ensamble()