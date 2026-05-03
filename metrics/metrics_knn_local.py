import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_knn_local(knn_model, train_labels, test_df, model_st, n_samples=100):
    """
    Evalúa el modelo k-NN localmente sin necesidad de OpenSearch.
    """
    print(f"Evaluando {n_samples} muestras de prueba...")
    y_true = []
    y_pred = []

    # Tomamos una muestra del set de test
    sample_df = test_df.sample(n=min(len(test_df), n_samples))
    
    # Generamos los vectores para los mensajes de prueba
    test_embeddings = model_st.encode(sample_df['log_message'].tolist())
    
    # Buscamos el vecino más cercano (k=1) en el modelo entrenado
    _, index = knn_model.kneighbors(test_embeddings)

    for i, idx_list in enumerate(index):
        # Etiqueta real del log de prueba
        real_label = sample_df.iloc[i]['label']
        # Predicción: etiqueta del vecino más cercano en el set de entrenamiento
        pred_label = train_labels[idx_list[0]]
        
        y_true.append(real_label)
        y_pred.append(pred_label)

    print("\n--- Informe de Clasificación ---")
    print(classification_report(y_true, y_pred))
    
    return y_true, y_pred

def generate_confusion_matrix_plot(y_true, y_pred, output_path="matriz_confusion_tfm.png"):
    """Genera la imagen de la matriz para la memoria del TFM."""
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión - knn Local')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Gráfica guardada en: {output_path}")