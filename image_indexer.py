# image_indexer.py

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import pickle
from tqdm import tqdm

# --- Configuration ---
# Note: These paths are now relative to the app's root
UPLOAD_FOLDER = "static/uploads"
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "image_index.faiss")
MAPPING_PATH = os.path.join(INDEX_DIR, "image_paths.pkl")
MODEL_ID = "openai/clip-vit-base-patch32"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def create_index():
    """
    Processes images in the upload folder, generates CLIP embeddings,
    and saves them into a FAISS index. Returns the number of images indexed.
    """
    print("Loading CLIP model for indexing...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = CLIPModel.from_pretrained(MODEL_ID).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return 0
    
    print(f"CLIP model loaded on {device}.")

    image_paths = [os.path.join(UPLOAD_FOLDER, fname) for fname in os.listdir(UPLOAD_FOLDER)
                   if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_paths:
        print("No images found in upload folder to index.")
        # If an old index exists, remove it
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(MAPPING_PATH):
            os.remove(MAPPING_PATH)
        return 0

    print(f"Found {len(image_paths)} images to index.")

    all_embeddings = []
    valid_image_paths = []
    
    for path in tqdm(image_paths, desc="Processing Images"):
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=inputs.pixel_values)
            all_embeddings.append(image_features.cpu().numpy().flatten())
            valid_image_paths.append(path)
        except Exception as e:
            print(f"Could not process {path}: {e}")

    if not all_embeddings:
        print("No embeddings were generated.")
        return 0

    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    embedding_dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    print(f"FAISS index built with {index.ntotal} vectors.")
    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, 'wb') as f:
        pickle.dump(valid_image_paths, f)

    print(f"Index and mapping saved successfully.")
    return index.ntotal
