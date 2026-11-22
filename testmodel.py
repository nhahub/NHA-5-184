import tensorflow as tf

model = tf.keras.models.load_model(
    "models/B_full_s42_384_mix_model.keras",
    compile=False
)

print("Model loaded successfully!")
model.summary()