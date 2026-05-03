import pandas as pd
import os

def apply_downsampling(input_csv, output_csv, n_normal_samples=50000):
    if not os.path.exists(input_csv):
        print(f"Error: No se encuentra {input_csv}")
        return

    print("Cargando dataset original...")
    df = pd.read_csv(input_csv)

    # 1. Separar ataques de logs normales
    ataques_df = df[df['label'] != 'normal_log']
    normal_df = df[df['label'] == 'normal_log']

    print(f"Registros de ataque encontrados: {len(ataques_df)}")
    print(f"Registros normales originales: {len(normal_df)}")

    # 2. Realizar el muestreo de los logs normales
    # Usamos un número fijo para tener un dataset manejable para el TFM
    if len(normal_df) > n_normal_samples:
        normal_downsampled = normal_df.sample(n=n_normal_samples, random_state=42)
    else:
        normal_downsampled = normal_df

    # 3. Combinar y barajar
    df_balanceado = pd.concat([ataques_df, normal_downsampled])
    df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. Guardar el nuevo dataset
    df_balanceado.to_csv(output_csv, index=False)
    
    print("\n--- Resultado del Downsampling ---")
    print(f"Nuevo total de registros: {len(df_balanceado)}")
    print(df_balanceado['label'].value_counts())
    print(f"Archivo guardado como: {output_csv}")

if __name__ == "__main__":
    INPUT = "logs-dataset-raw.csv"
    OUTPUT = "logs-dataset-reduced.csv"
    
    # En el TFG se recomienda reducir a una cantidad manejable 
    # para que el entrenamiento del LLM/Red Neuronal no sea eterno.
    apply_downsampling(INPUT, OUTPUT, n_normal_samples=50000)