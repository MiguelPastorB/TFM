import pandas as pd
import os

def apply_downsampling(input_csv, output_csv, n_normal_samples=50000):
    if not os.path.exists(input_csv):
        print(f"Error: No se encuentra {input_csv}")
        return

    print("Cargando dataset original...")
    df = pd.read_csv(input_csv)

    # Separamos ataques de logs normales
    attacks_df = df[df['label'] != 'normal_log']
    normal_df = df[df['label'] == 'normal_log']

    print(f"Registros de ataque encontrados: {len(attacks_df)}")
    print(f"Registros normales originales: {len(normal_df)}")

    # Realizamos el muestreo de los logs normales
    if len(normal_df) > n_normal_samples:
        normal_downsampled = normal_df.sample(n=n_normal_samples, random_state=42)
    else:
        normal_downsampled = normal_df

    # Concatenamos los ataques con los logs normales muestreados
    df_reduced = pd.concat([attacks_df, normal_downsampled])
    df_reduced = df_reduced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Guardamos el nuevo dataset
    df_reduced.to_csv(output_csv, index=False)
    
    print("\n--- Resultado del Downsampling ---")
    print(f"Nuevo total de registros: {len(df_reduced)}")
    print(df_reduced['label'].value_counts())
    print(f"Archivo guardado como: {output_csv}")

if __name__ == "__main__":
    INPUT = "logs-dataset-raw.csv"
    OUTPUT = "logs-dataset-reduced.csv"
    
    # Seleccionamos el número de muestras normales a mantener
    apply_downsampling(INPUT, OUTPUT, n_normal_samples=50000)