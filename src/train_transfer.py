import pandas as pd
import numpy as np
import os
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split
from mlflow_utils import setup_mlflow

# --- Chemins ---
INPUT_TRAIN = os.path.join("data", "fashion-mnist", "fashion-mnist_train.csv")
MODEL_OUTPUT = os.path.join("models", "transfer_model.keras")

TARGET_SIZE = (64, 64)
NUM_CLASSES = 10
BATCH_SIZE = 256   # très important (256 = optimal pour ton PC)


class FashionGenerator(Sequence):
    """Génère les images batch par batch pour éviter le crash mémoire."""
    def __init__(self, df, batch_size=BATCH_SIZE, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(df))
        self.on_epoch_end()

    def __len__(self):
        return len(self.df) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_ids = self.indices[index*self.batch_size : (index+1)*self.batch_size]

        batch = self.df.iloc[batch_ids]

        X = batch.drop("label", axis=1).values.reshape(-1, 28, 28, 1)
        X = tf.image.resize(X, TARGET_SIZE)
        X = tf.concat([X, X, X], axis=-1)
        X = preprocess_input(X)

        y = to_categorical(batch["label"].values, NUM_CLASSES)

        return X.numpy(), y


def train_transfer(train_path, model_output_path):
    print("--- Train Transfer (64×64, Generator, Stable) ---")
    
    setup_mlflow()
    mlflow.set_experiment("Fashion_MNIST_Transfer")

    with mlflow.start_run():
        df = pd.read_csv(train_path)
    
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
        train_gen = FashionGenerator(train_df, BATCH_SIZE)
        val_gen = FashionGenerator(val_df, BATCH_SIZE, shuffle=False)
    
        # MobileNetV2
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(64, 64, 3),
            alpha=0.5
        )
    
        base_model.trainable = False  # phase 1
    
        inputs = Input(shape=(64, 64, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    
        model = Model(inputs, outputs)
    
        # Phase 1 — entraînement du classifieur
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        
        mlflow.log_param("epochs", 5) # Phase 1
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("model_type", "Transfer_MobileNetV2")
    
        model.fit(train_gen, validation_data=val_gen, epochs=5)
    
        # Débloquer les 20 dernières couches pour fine-tuning
        for layer in base_model.layers[-20:]:
            layer.trainable = True
    
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    
        # Phase 2 — fine-tuning
        history = model.fit(train_gen, validation_data=val_gen, epochs=5)
        
        mlflow.log_metric("accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("loss", history.history['loss'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
    
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        model.save(model_output_path)
        
        # Log model file as artifact
        mlflow.log_artifact(model_output_path, artifact_path="models")
        
        # Log training history as artifact
        import json
        history_path = "training_history_transfer.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        mlflow.log_artifact(history_path)
        os.remove(history_path)
        
        # Log model summary as artifact
        summary_path = "model_summary_transfer.txt"
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact(summary_path)
        os.remove(summary_path)
        
        # Log model to MLflow registry
        mlflow.tensorflow.log_model(model, "model", registered_model_name="Fashion_MNIST_Transfer")
    
        print(f"Modèle sauvegardé : {model_output_path}")
        print("--- Training terminé (aucun crash) ---")


if __name__ == "__main__":
    train_transfer(INPUT_TRAIN, MODEL_OUTPUT)
