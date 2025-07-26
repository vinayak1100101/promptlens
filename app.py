# app.py

import os
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import pickle
import google.generativeai as genai
from image_indexer import create_index, UPLOAD_FOLDER, INDEX_PATH, MAPPING_PATH

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global Variables & Model Loading ---
# The API key will be stored in memory for the duration of the app's life
SESSION_API_KEY = None
GEMINI_MODEL = None

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model once on startup
print("Loading CLIP model for search...")
try:
    CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded successfully.")
except Exception as e:
    CLIP_MODEL, CLIP_PROCESSOR = None, None
    print(f"Fatal Error: Could not load CLIP model. The app will not work. {e}")

# --- Route Definitions ---
@app.route('/')
def index():
    """Renders the main page."""
    api_key_set = bool(SESSION_API_KEY)
    return render_template('index.html', api_key_set=api_key_set)

@app.route('/save_api_key', methods=['POST'])
def save_api_key():
    """Saves the Google API key in memory and validates it."""
    global SESSION_API_KEY, GEMINI_MODEL
    
    data = request.get_json()
    api_key = data.get('api_key')
    if not api_key:
        return jsonify({'status': 'error', 'message': 'API key is missing.'}), 400
    
    # Try to configure Gemini with the new key to validate it
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        # Make a test call
        model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1))
        
        # If successful, save the key and model instance
        SESSION_API_KEY = api_key
        GEMINI_MODEL = model
        print("Gemini API key validated and saved to session.")
        return jsonify({'status': 'success', 'message': 'API key is valid and saved for this session.'})
    except Exception as e:
        # If it fails, reset the key
        SESSION_API_KEY = None
        GEMINI_MODEL = None
        print(f"API Key validation failed: {e}")
        return jsonify({'status': 'error', 'message': 'Invalid API Key or configuration error.'}), 400

@app.route('/upload_images', methods=['POST'])
def upload_images():
    """Handles image uploads and triggers re-indexing."""
    if 'images' not in request.files:
        return jsonify({'status': 'error', 'message': 'No images part in the request.'}), 400
    
    files = request.files.getlist('images')
    for file in files:
        if file.filename == '':
            continue
        if file:
            # Note: In a real-world app, you'd want to sanitize filenames
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    try:
        count = create_index()
        return jsonify({'status': 'success', 'message': f'Images uploaded and indexed. Total images: {count}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error during indexing: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search():
    """Performs the image search."""
    if not CLIP_MODEL:
         return jsonify({'error': 'CLIP model not loaded.'}), 500

    if not os.path.exists(INDEX_PATH):
        return jsonify({'error': 'No index found. Please upload images first.'}), 400

    data = request.get_json()
    query = data.get('query')
    enhance = data.get('enhance', False)
    
    enhanced_query_text = None
    if enhance:
        if not GEMINI_MODEL:
            return jsonify({'error': 'Gemini API key not set or invalid. Cannot enhance prompt.'}), 400
        
        try:
            system_instruction = (
                "You are an expert in visual descriptions. Expand the user's simple image search query "
                "into a rich, descriptive sentence. Focus on objects, scenery, colors, and mood. "
                "Do not respond with anything but the enhanced prompt itself."
            )
            response = GEMINI_MODEL.generate_content(f"{system_instruction}\n\nUser query: \"{query}\"")
            enhanced_query_text = response.text.strip()
            query = enhanced_query_text # Use the enhanced query for the search
        except Exception as e:
            return jsonify({'error': f'Error calling Gemini API: {str(e)}'}), 500

    # Load index and mapping for each search
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, 'rb') as f:
        image_paths = pickle.load(f)

    # Embed text query
    inputs = CLIP_PROCESSOR(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_embedding = CLIP_MODEL.get_text_features(**inputs).cpu().numpy()

    # Search FAISS
    distances, indices = index.search(text_embedding.astype(np.float32), k=9)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            similarity = 1 / (1 + distances[0][i])
            results.append({
                "path": image_paths[idx],
                "score": f"{similarity:.4f}"
            })
            
    return jsonify({'results': results, 'enhanced_query': enhanced_query_text})

if __name__ == '__main__':
    app.run(debug=True)
