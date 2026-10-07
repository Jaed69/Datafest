import json
import pandas as pd

def hacer_ensamble_train():
    print("Buscando las predicciones seguras (Out-of-Fold) en index.json...")
    with open('experiments/index.json', 'r') as f:
        d = json.load(f)
    
    # Filtrar las mejores corridas
    lgb_runs = [r for r in d['runs'] if 'lightgbm' in r['model_name']]
    cat_runs = [r for r in d['runs'] if 'catboost' in r['model_name']]
    mejor_lgb = sorted(lgb_runs, key=lambda x: x['inner_mean_gini'], reverse=True)[0]
    mejor_cat = sorted(cat_runs, key=lambda x: x['inner_mean_gini'], reverse=True)[0]
    
    # Rutas a los archivos OOF
    val_lgb_path = mejor_lgb['run_dir'] + '/validation_predictions.csv'
    val_cat_path = mejor_cat['run_dir'] + '/validation_predictions.csv'
    
    df_lgb = pd.read_csv(val_lgb_path)
    df_cat = pd.read_csv(val_cat_path)
    
    # Detectar el nombre de la columna de predicción
    col_proba = df_lgb.columns[-1] if 'prediccion' not in df_lgb.columns else 'prediccion'
    
    # Crear el ensamble promediando las probabilidades
    df_ensamble = df_lgb.copy()
    df_ensamble[col_proba] = (df_lgb[col_proba] * 0.6) + (df_cat[col_proba] * 0.4)
    
    # Filtrar para mantener solo id_cliente y la predicción
    if 'id_cliente' in df_ensamble.columns:
        df_ensamble = df_ensamble[['id_cliente', col_proba]]
        # Si un cliente aparece en múltiples meses de validación, promediamos su score
        df_ensamble = df_ensamble.groupby('id_cliente', as_index=False).mean()
        
    df_ensamble.rename(columns={col_proba: 'prediccion'}, inplace=True)
    df_ensamble.to_csv('ENTREGABLE_TRAIN.csv', index=False)
    
    print(f"\n¡Éxito! El archivo se guardó como: ENTREGABLE_TRAIN.csv")
    print(f"Total de clientes únicos puntuados: {len(df_ensamble)}")

if __name__ == "__main__":
    hacer_ensamble_train()
