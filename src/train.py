import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from mlflow_utils import setup_mlflow

INPUT_TRAIN = os.path.join('data', 'fashion-mnist', 'fashion-mnist_train.csv')
INPUT_TEST = os.path.join('data', 'fashion-mnist', 'fashion-mnist_test.csv')
MODEL_OUTPUT = os.path.join('models', 'fashion_classifier.keras')

def train_model(train_path, test_path, model_output_path):
    """
    Charge les données, entraîne un réseau de neurones simple, et sauvegarde le modèle.
    """
    print("--- Démarrage de l'entraînement du modèle ---")
    
    setup_mlflow()
    mlflow.set_experiment("Fashion_MNIST_Simple_MLP")

    with mlflow.start_run():
        # 1. Chargement des données prétraitées
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        X_train = df_train.drop('label', axis=1).values
        y_train = df_train['label'].values
        X_test = df_test.drop('label', axis=1).values
        y_test = df_test['label'].values
        
        # Le dataset est déjà normalisé (si vous l'aviez fait manuellement, sinon ajouter ici une étape de normalisation si nécessaire)
        # Assurez-vous que les valeurs sont entre 0 et 1.
        
        # 2. Encodage One-Hot des labels (nécessaire pour la classification multi-classes dans Keras)
        num_classes = 10
        y_train_encoded = to_categorical(y_train, num_classes=num_classes)
        y_test_encoded = to_categorical(y_test, num_classes=num_classes)
    
        # 3. Définition du modèle (Réseau de Neurones Simple)
        model = Sequential([
            Dense(512, activation='relu', input_shape=(784,)),
            Dense(256, activation='relu'),
            Dense(num_classes, activation='softmax') # La couche de sortie utilise softmax
        ])
    
        # 4. Compilation et Entraînement
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
    
        epochs = 5
        batch_size = 256
        
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model_type", "Simple_MLP")
        
        # Entraînement sur un petit nombre d'epochs pour la rapidité
        history = model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Log metrics
        loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)
    
        # 5. Création du dossier de sortie et Sauvegarde du modèle
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        model.save(model_output_path)
        
        # Log model file as artifact
        mlflow.log_artifact(model_output_path, artifact_path="models")
        
        # Log training history as artifact
        import json
        history_path = "training_history_mlp.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        mlflow.log_artifact(history_path)
        os.remove(history_path)
        
        # Log model summary as artifact
        summary_path = "model_summary_mlp.txt"
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact(summary_path)
        os.remove(summary_path)
        
        # Log model to MLflow registry
        mlflow.tensorflow.log_model(model, "model", registered_model_name="Fashion_MNIST_Simple_MLP")
    
        print(f"Modèle sauvegardé avec succès à : {model_output_path}")
        print("--- Entraînement terminé ---")

if __name__ == "__main__":
    train_model(INPUT_TRAIN, INPUT_TEST, MODEL_OUTPUT)