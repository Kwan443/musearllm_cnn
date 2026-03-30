import os
import sys
import json
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class MultiArtworkFeatureExtractor:
    def __init__(self, model_path="../resnet-50", storage_dir="../artwork_features_multi"):
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

    def group_images_by_prefix(self, images_dir):
        """Group images by their prefix (p1, p2, etc.)"""
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        grouped_images = {}
        for image_file in image_files:
            # Extract prefix (p1, p2, etc.)
            prefix = image_file.split(' ')[0]  # Gets 'p1', 'p2', etc.
            if prefix not in grouped_images:
                grouped_images[prefix] = []
            
            full_path = os.path.join(images_dir, image_file)
            grouped_images[prefix].append(full_path)
        
        print(f"📁 Found {len(grouped_images)} artwork groups:")
        for prefix, paths in grouped_images.items():
            print(f"   {prefix}: {len(paths)} photos")
        
        return grouped_images

    def save_multi_photo_features(self, image_paths, artwork_id):
        """Save multiple photos for a single artwork"""
        all_features = []
        successful_photos = []
        
        for i, image_path in enumerate(image_paths):
            print(f"  📸 Processing photo {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            features = self.extract_single_features(image_path)
            if features is not None:
                all_features.append(features)
                successful_photos.append(image_path)
        
        if not all_features:
            print(f"❌ No features extracted for {artwork_id}")
            return False
        
        # Save multiple feature vectors for the same artwork
        feature_paths = []

        # If artwork already exists in metadata, append new photos instead of overwriting
        existing_meta = self.metadata.get(artwork_id)
        start_index = 0
        existing_feature_paths = []
        existing_photo_paths = []
        if existing_meta:
            existing_photo_paths = existing_meta.get('photo_paths', [])
            existing_feature_paths = existing_meta.get('feature_paths', [])
            start_index = existing_meta.get('photo_count', 0)

        for i, features in enumerate(all_features):
            idx = start_index + i + 1
            feature_filename = f"{artwork_id}_photo{idx}.npy"
            feature_path = os.path.join(self.features_dir, feature_filename)
            # Ensure directory exists (defensive)
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            try:
                np.save(feature_path, features)
                feature_paths.append(feature_path)
            except Exception as e:
                print(f"   ⚠️ Failed to save feature {feature_path}: {e}")

        # Merge with existing metadata if present
        merged_photo_paths = list(existing_photo_paths) + successful_photos
        merged_feature_paths = list(existing_feature_paths) + feature_paths

        self.metadata[artwork_id] = {
            'artwork_id': artwork_id,
            'photo_count': len(merged_feature_paths),
            'photo_paths': merged_photo_paths,
            'feature_paths': merged_feature_paths,
            'timestamp': np.datetime64('now').astype(str)
        }
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Saved {len(all_features)} photos for artwork: {artwork_id}")
        return True

    def process_multi_photo_directory(self, images_dir):
        """Process directory with multiple photos per artwork"""
        grouped_images = self.group_images_by_prefix(images_dir)
        
        total_processed = 0
        
        for artwork_id, photo_paths in grouped_images.items():
            # If artwork exists in metadata, check whether all photos already processed
            existing = self.metadata.get(artwork_id)
            if existing:
                existing_basenames = {os.path.basename(p) for p in existing.get('photo_paths', [])}
                new_paths = [p for p in photo_paths if os.path.basename(p) not in existing_basenames]
                if not new_paths:
                    print(f"⏭️ Skipping {artwork_id} (already processed)")
                    continue
                print(f"\n🖼️ Processing artwork (adding {len(new_paths)} new photos): {artwork_id}")
                success = self.save_multi_photo_features(new_paths, artwork_id)
            else:
                print(f"\n🖼️ Processing artwork: {artwork_id}")
                success = self.save_multi_photo_features(photo_paths, artwork_id)

            if success:
                total_processed += 1
        
        print(f"\n🎉 Multi-photo processing completed!")
        print(f"   Processed {total_processed} artworks")
        print(f"   Total photos: {sum(len(paths) for paths in grouped_images.values())}")

def main():
    # Initialize extractor
    extractor = MultiArtworkFeatureExtractor()
    
    # Process the test_multi directory
    test_multi_dir = "../photos"
    
    if os.path.exists(test_multi_dir):
        extractor.process_multi_photo_directory(test_multi_dir)
    else:
        print(f"❌ Directory not found: {test_multi_dir}")

if __name__ == "__main__":
    main()