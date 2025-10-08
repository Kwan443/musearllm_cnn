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
        for i, features in enumerate(all_features):
            feature_filename = f"{artwork_id}_photo{i+1}.npy"
            feature_path = os.path.join(self.features_dir, feature_filename)
            np.save(feature_path, features)
            feature_paths.append(feature_path)
        
        # Update metadata with multiple photos
        self.metadata[artwork_id] = {
            'artwork_id': artwork_id,
            'photo_count': len(all_features),
            'photo_paths': successful_photos,
            'feature_paths': feature_paths,
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
            # Skip if already processed
            if artwork_id in self.metadata:
                print(f"⏭️ Skipping {artwork_id} (already processed)")
                continue
            
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
    test_multi_dir = "../test_multi"
    
    if os.path.exists(test_multi_dir):
        extractor.process_multi_photo_directory(test_multi_dir)
    else:
        print(f"❌ Directory not found: {test_multi_dir}")

if __name__ == "__main__":
    main()