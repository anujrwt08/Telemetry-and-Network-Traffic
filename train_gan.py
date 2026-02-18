import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import os

# --- CONFIGURATION ---
LATENT_DIM = 100       # Size of the random noise vector (paper uses 100)
BATCH_SIZE = 64        # Batch size for training
EPOCHS = 2000          # Training loops (2000 is needed for small data like mitm)
TARGET_CLASS = 'mitm'  # The minority class we want to fix
TARGET_COUNT = 20000   # We want to reach this number of samples

# 1. Load Processed Data
print("Loading processed data...")
if not os.path.exists('processed_network_data.csv'):
    print("Error: 'processed_network_data.csv' not found. Run preprocess.py first.")
    exit()
    
df = pd.read_csv('processed_network_data.csv')

# 2. Isolate the Minority Class (MITM)
# We only train the GAN on the class we want to generate
minority_data = df[df['type'] == TARGET_CLASS].drop(columns=['label', 'type'])
data_dim = minority_data.shape[1]

print(f"\nTraining GAN on '{TARGET_CLASS}' class only.")
print(f"Real samples available: {len(minority_data)}")
print(f"Features per sample: {data_dim}")

# Convert to Float32 (TensorFlow requirement)
X_train = minority_data.values.astype('float32')

# --- 3. BUILD THE MODELS (Paper Architecture) ---

def build_generator():
    model = models.Sequential()
    # Input: Noise Vector (100) -> Hidden Layers -> Output Features
    model.add(layers.Dense(128, input_dim=LATENT_DIM))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    
    # Output: Features (Sigmoid because our data is scaled 0-1)
    model.add(layers.Dense(data_dim, activation='sigmoid'))
    return model

def build_discriminator():
    model = models.Sequential()
    # Input: Actual Features -> Hidden Layers -> Real/Fake Decision
    model.add(layers.Dense(512, input_dim=data_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # Output: Real (1) or Fake (0)
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

# Initialize Models
discriminator = build_discriminator()
generator = build_generator()

# Combined model (Stacked Generator + Discriminator)
discriminator.trainable = False  # Freeze discriminator when training generator
z = layers.Input(shape=(LATENT_DIM,))
img = generator(z)
valid = discriminator(img)
combined = models.Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# --- 4. TRAINING LOOP ---
print(f"\nStarting Training for {EPOCHS} epochs... (This may take 5-10 mins)")
start_time = time.time()

valid_labels = np.ones((BATCH_SIZE, 1))
fake_labels = np.zeros((BATCH_SIZE, 1))

for epoch in range(EPOCHS):
    # -- Train Discriminator --
    # Select random real samples
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    imgs = X_train[idx]
    
    # Generate fake samples
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    gen_imgs = generator.predict(noise, verbose=0)
    
    # Train
    d_loss_real = discriminator.train_on_batch(imgs, valid_labels)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # -- Train Generator --
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = combined.train_on_batch(noise, valid_labels) # We want discriminator to call them "valid"

    # Print progress every 200 epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch}/{EPOCHS} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

print(f"Training finished in {time.time() - start_time:.2f} seconds.")

# --- 5. GENERATE SYNTHETIC DATA ---
samples_needed = TARGET_COUNT - len(minority_data)
print(f"\nGenerating {samples_needed} synthetic '{TARGET_CLASS}' samples...")

noise = np.random.normal(0, 1, (samples_needed, LATENT_DIM))
synthetic_data = generator.predict(noise)

# Convert to DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=minority_data.columns)
synthetic_df['type'] = TARGET_CLASS
synthetic_df['label'] = 1 # Mark as Attack (1)

# --- 6. MERGE & SAVE ---
print("Merging with original dataset...")
balanced_df = pd.concat([df, synthetic_df], axis=0)

# Verify new counts
print("\n--- NEW CLASS DISTRIBUTION ---")
print(balanced_df['type'].value_counts())

print(f"\nSaving to 'balanced_network_data.csv'...")
balanced_df.to_csv('balanced_network_data.csv', index=False)
print("Done! Phase 2 Complete.")