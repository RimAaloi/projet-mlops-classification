import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- Chemins de Fichiers & Modèles ---
INPUT_TEST = os.path.join('data', 'fashion-mnist', 'fashion-mnist_test.csv')
METRICS_OUTPUT = os.path.join('metrics', 'metrics.json')
PLOTS_DIR = os.path.join('metrics', 'plots')

# Liste des modèles à évaluer
MODEL_PATHS = {
    'simple_mlp': os.path.join('models', 'fashion_classifier.keras'),
    'cnn': os.path.join('models', 'cnn_model.keras'),
    'transfer_learning': os.path.join('models', 'transfer_model.keras')
}
TARGET_SIZE = (64, 64) # Taille utilisée pour le Transfer Learning
LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_and_preprocess_test_data(X_test, model_type):
    """Charge et prétraite les données de test selon le type de modèle requis."""
    
    if model_type == 'transfer_learning':
        # Reshape 28x28x1 -> 64x64x3 pour MobileNetV2
        X_images = X_test.reshape(-1, 28, 28)
        X_resized = tf.image.resize(X_images[..., tf.newaxis], TARGET_SIZE)
        X_rgb = tf.concat([X_resized, X_resized, X_resized], axis=-1)
        # Normalisation spécifique à MobileNetV2
        X_processed = mobilenet_preprocess(X_rgb).numpy()
    elif model_type == 'cnn':
        # Reshape 28x28x1 pour CNN
        X_processed = X_test.reshape(-1, 28, 28, 1)
    else: # simple_mlp
        # Pas de reshape (déjà plat 784 pixels)
        X_processed = X_test
    
    return X_processed

def generate_confusion_matrix(y_test, y_pred, model_name, plot_dir):
    """Génère et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f'Matrice de Confusion: {model_name}')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def evaluate_and_compare(test_path, model_paths, metrics_output_path, plot_dir):
    """Évalue et compare tous les modèles définis dans MODEL_PATHS."""
    print("--- Démarrage de l'évaluation et de la comparaison des modèles ---")
    results = {}

    # Chargement des données de test brutes une seule fois
    df_test = pd.read_csv(test_path)
    X_test_raw = df_test.drop('label', axis=1).values
    y_test = df_test['label'].values

    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"ATTENTION : Modèle {name} non trouvé à {path}. Le sauter.")
            continue
            
        print(f"\nÉvaluation du modèle : {name}...")
        
        # 1. Prétraitement spécifique au modèle
        X_test_processed = load_and_preprocess_test_data(X_test_raw, name)
        
        # 2. Chargement et Prédiction
        model = load_model(path)
        y_pred_probs = model.predict(X_test_processed)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # 3. Calcul et enregistrement des métriques
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 4. Génération des visualisations
        cm_plot_path = generate_confusion_matrix(y_test, y_pred, name, plot_dir)

        # 5. Enregistrement des résultats
        results[name] = {
            'accuracy': float(f'{acc:.4f}'),
            'f1_score': float(f'{f1:.4f}'),
            'confusion_matrix_plot': cm_plot_path # Ceci sera utilisé pour le rapport
        }
        print(f"Résultats pour {name}: Accuracy {acc:.4f}, F1-Score {f1:.4f}")

    # 6. Sauvegarde du fichier JSON final
    with open(metrics_output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nRésultats de la comparaison sauvegardés dans : {metrics_output_path}")
    print("--- Évaluation et comparaison terminées ---")

if __name__ == "__main__":
    evaluate_and_compare(INPUT_TEST, MODEL_PATHS, METRICS_OUTPUT, PLOTS_DIR)