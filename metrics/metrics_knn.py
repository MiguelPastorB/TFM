import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Importamos la función de inferencia que ya definiste[cite: 3]
from knn.inference_knn import knn_rag

def evaluate_knn_classification(client, test_df, index_name, knn_func, n_samples=100):
    """
    Evalúa el rendimiento comparando etiquetas reales con predicciones de OpenSearch.
    """
    print(f"Evaluando rendimiento con {n_samples} muestras...")
    y_true = []
    y_pred = []

    # Seleccionar muestra aleatoria para la evaluación[cite: 1]
    sample_df = test_df.sample(n=min(len(test_df), n_samples))

    for _, row in sample_df.iterrows():
        # Llamada a la función de inferencia k-NN[cite: 3]
        outcomes = knn_func(client, row['log_message'], index_name, k=1)
        
        y_true.append(row['label'])
        if outcomes:
            y_pred.append(outcomes[0]['label'])
        else:
            y_pred.append("unknown")

    print("\n--- Informe de Clasificación ---")
    print(classification_report(y_true, y_pred))
    
    return y_true, y_pred

def generate_confusion_matrix_plot(y_true, y_pred, output_path="resultado_knn.png"):
    """Genera una matriz de confusión visual"""
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title('Matriz de Confusión: Clasificación de Logs vía k-NN')
    plt.ylabel('Etiqueta Real (Ground Truth)')
    plt.xlabel('Predicción (OpenSearch k-NN)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Gráfica de rendimiento guardada en: {output_path}")
    plt.show()