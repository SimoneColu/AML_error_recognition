import numpy as np
import os

# carica il primo file dalla cartella step_embeddings
step_embeddings_dir = "step_embeddings"
files = sorted(os.listdir(step_embeddings_dir))
first_file = files[0]
filepath = os.path.join(step_embeddings_dir, first_file)

print(f"File: {first_file}")
print("=" * 60)

# carica il file npz
data = np.load(filepath)

print(f"\nChiavi disponibili: {list(data.keys())}")
print(f"Numero di step: {len(data['step_ids'])}")

print("\n--- Step IDs ---")
print(data['step_ids'])

print("\n--- Has Errors ---")
print(data['has_errors'])

print("\n--- Tempi (start -> end) ---")
for i, (start, end) in enumerate(zip(data['start_times'], data['end_times'])):
    duration = end - start
    print(f"  Step {i:2d} (id={data['step_ids'][i]:3d}): {start:8.2f}s -> {end:8.2f}s  (durata: {duration:.2f}s)")

print("\n--- Embeddings ---")
print(f"Shape: {data['embeddings'].shape}")
print(f"Dtype: {data['embeddings'].dtype}")
print("\nPrimi 3 embedding (primi 15 valori):")
for i in range(min(3, len(data['embeddings']))):
    emb = data['embeddings'][i][:15]
    print(f"  Step {i}: {emb}")
