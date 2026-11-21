# =====================================================================
# Stage 1 (DR vs No DR) — TEAM-WRITTEN reference code (kept verbatim in logic).
# Trained on Kaggle on the preprocessed APTOS arrays. Output: best_binary_dr_model.keras
# (frozen as stage1_v1.keras). Reported: acc ~0.95, DR recall ~0.94 @0.5 / ~0.95 @0.3 on APTOS val.
# Known limitations (for the reviewer): no global seed; default threshold 0.5 (0.3 chosen later);
# validation reused for checkpointing + threshold choice; images are multiplied by 255 inside the
# tf.data pipeline because EfficientNet (Keras) normalizes internally.
# =====================================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "/kaggle/input/datasets/omarashour206/aptos2019-preprocessed-dataset"
MODEL_PATH = "/kaggle/working/best_binary_dr_model.keras"
BATCH_SIZE = 32
EPOCHS = 10

# ---- quick look at the data ----
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"), mmap_mode="r")
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.int32)
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"), mmap_mode="r")
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.int32)
print("X_train:", X_train.shape, X_train.dtype, "| X_val:", X_val.shape)
print(pd.DataFrame({"Train": np.bincount(y_train, minlength=5), "Val": np.bincount(y_val, minlength=5)},
                   index=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]))

print("GPU:", tf.config.list_physical_devices("GPU"))

# ---- binary labels ----
y_train = (y_train > 0).astype(np.float32)
y_val = (y_val > 0).astype(np.float32)
print("Train:", X_train.shape, np.bincount(y_train.astype(int)))
print("Validation:", X_val.shape, np.bincount(y_val.astype(int)))


def make_dataset(X, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((np.arange(len(y)), y))
    if training:
        ds = ds.shuffle(len(y))

    def load_batch(indices, labels):
        images = tf.numpy_function(
            lambda idx: np.asarray(X[idx], dtype=np.float32) * 255.0,
            [indices], tf.float32)
        images.set_shape([None, 224, 224, 3])
        labels = tf.cast(labels, tf.float32)
        return images, labels

    ds = ds.batch(BATCH_SIZE).map(load_batch, num_parallel_calls=1)
    return ds.prefetch(1)


train_ds = make_dataset(X_train, y_train, training=True)
val_ds = make_dataset(X_val, y_val)

weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train.astype(int))
class_weights = {0: float(weights[0]), 1: float(weights[1])}
print("Class weights:", class_weights)

base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True, verbose=1),
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights,
          callbacks=callbacks, verbose=1)

model = tf.keras.models.load_model(MODEL_PATH)
probabilities = model.predict(val_ds, verbose=1).ravel()
predictions = (probabilities >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_val.astype(int), predictions, labels=[0, 1],
                            target_names=["No DR", "DR"], zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_val.astype(int), predictions))
print("\nSaved model:", MODEL_PATH)
