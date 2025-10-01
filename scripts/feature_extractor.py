import os
import sys
import json
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch

# Add the parent directory to path so we can import other modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class ArtworkFeatureExtractor:
    def __init__(self, model_path="../resnet-50", storage_dir="../artwork_features"):
        self.model_path = model_path
        self.storage_dir = storage_dir
        self.features_dir = os.path.join(storage_dir, "features")
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        
        # Create directories
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Load model
        print("Loading model...")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        print("Model loaded successfully!")
        
        # Load or initialize metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def extract_single_features(self, image_path):
        """Extract features from a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use pooler output or last hidden state
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output.squeeze().cpu().numpy()
            else:
                features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
            return features
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def save_features(self, image_path, features, artwork_id):
        """Save features to local storage"""
        # Save features as .npy file
        feature_filename = f"{artwork_id}.npy"
        feature_path = os.path.join(self.features_dir, feature_filename)
        np.save(feature_path, features)
        
        # Update metadata
        self.metadata[artwork_id] = {
            'feature_path': feature_path,
            'image_path': image_path,
            'artwork_id': artwork_id,
            'feature_dim': len(features),
            'timestamp': np.datetime64('now').astype(str)
        }
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Saved features for {artwork_id}")

    def process_directory(self, images_dir, file_extensions=('.jpg', '.jpeg', '.png')):
        """Process all images in a directory"""
        image_files = []
        
        for file in os.listdir(images_dir):
            if file.lower().endswith(file_extensions):
                image_files.append(os.path.join(images_dir, file))
        
        print(f"Found {len(image_files)} images to process")
        
        for i, image_path in enumerate(image_files):
            # Generate artwork ID from filename
            artwork_id = os.path.splitext(os.path.basename(image_path))[0]
            
            # Skip if already processed
            if artwork_id in self.metadata:
                print(f"⏭️ Skipping {artwork_id} (already processed)")
                continue
            
            print(f"Processing {i+1}/{len(image_files)}: {artwork_id}")
            
            features = self.extract_single_features(image_path)
            if features is not None:
                self.save_features(image_path, features, artwork_id)

def main():
    # Initialize extractor
    extractor = ArtworkFeatureExtractor()
    
    # Process all images in the artworks directory
    extractor.process_directory("../artworks")
    
    print("🎉 Feature extraction completed!")
    print(f"Processed {len(extractor.metadata)} artworks")

if __name__ == "__main__":
    main()