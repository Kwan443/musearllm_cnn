import os
import numpy as np
import json
from multi_feature_extractor import MultiArtworkFeatureExtractor

class MultiPhotoSimilaritySearch:
    def __init__(self, storage_dir="../artwork_features_multi"):
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self.features_dir = os.path.join(storage_dir, "features")
        
        # Load metadata (be tolerant if file missing)
        if not os.path.exists(self.metadata_file):
            print(f"⚠️ Metadata file not found: {self.metadata_file} — continuing with empty index")
            self.metadata = {}
        else:
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load metadata.json: {e} — continuing with empty metadata")
                self.metadata = {}
        
        # Load features for multi-photo artworks
        self.artwork_features = {}  # {artwork_id: [feature1, feature2, ...]}
        self.artwork_ids = []
        
        for artwork_id, meta in self.metadata.items():
            feature_paths = meta.get('feature_paths', [])
            features_list = []

            for feature_path in feature_paths:
                try:
                    if os.path.exists(feature_path):
                        features = np.load(feature_path)
                        features_list.append(features)
                    else:
                        print(f"   ⚠️ Feature file missing: {feature_path}")
                except Exception as e:
                    print(f"   ⚠️ Failed to load feature {feature_path}: {e}")
            
            if features_list:
                # Normalize each feature vector
                normalized_features = []
                for features in features_list:
                    norm = np.linalg.norm(features)
                    if norm > 0:
                        normalized_features.append(features / norm)
                
                self.artwork_features[artwork_id] = normalized_features
                self.artwork_ids.append(artwork_id)
        
        print(f"✅ Loaded {len(self.artwork_ids)} artworks with multi-photo support")
        print(f"📊 Photo distribution:")
        for artwork_id, features in self.artwork_features.items():
            print(f"   {artwork_id}: {len(features)} photos")

    def compare_with_artwork(self, query_features, artwork_id):
        """Compare query with all photos of a specific artwork"""
        if artwork_id not in self.artwork_features:
            return None
        
        artwork_features = self.artwork_features[artwork_id]
        similarities = []
        
        # Normalize query features
        query_norm = query_features / np.linalg.norm(query_features)
        
        # Compare with each photo of the artwork
        for i, art_features in enumerate(artwork_features):
            similarity = np.dot(query_norm, art_features)
            similarities.append((i+1, similarity))  # (photo_number, similarity)
        
        return similarities

    def search_across_all_artworks(self, query_features, top_k=5):
        """Search across all artworks and their multiple photos"""
        query_norm = query_features / np.linalg.norm(query_features)
        artwork_scores = []
        
        for artwork_id, features_list in self.artwork_features.items():
            # Calculate similarity with each photo
            photo_similarities = []
            for features in features_list:
                similarity = np.dot(query_norm, features)
                photo_similarities.append(similarity)
            
            # Use the BEST similarity among all photos
            best_similarity = max(photo_similarities)
            avg_similarity = np.mean(photo_similarities)
            min_similarity = min(photo_similarities)
            
            artwork_scores.append({
                'artwork_id': artwork_id,
                'best_similarity': best_similarity,
                'avg_similarity': avg_similarity,
                'min_similarity': min_similarity,
                'photo_count': len(features_list),
                'all_similarities': photo_similarities
            })
        
        # Sort by best similarity
        artwork_scores.sort(key=lambda x: x['best_similarity'], reverse=True)
        return artwork_scores[:top_k]

    def detailed_artwork_comparison(self, query_image_path, extractor, target_artwork_id):
        """Get detailed comparison with a specific artwork's multiple photos"""
        print(f"\n🔍 Detailed comparison with {target_artwork_id}")
        print("=" * 50)
        
        query_features = extractor.extract_single_features(query_image_path)
        if query_features is None:
            return None
        
        # Get similarities with all photos of the target artwork
        photo_similarities = self.compare_with_artwork(query_features, target_artwork_id)
        
        if photo_similarities:
            print(f"🎨 Artwork: {target_artwork_id}")
            print(f"📸 Number of photos: {len(photo_similarities)}")
            print("\n📊 Similarity with each photo:")
            
            for photo_num, similarity in photo_similarities:
                percentage = similarity * 100
                stars = "★" * min(int(percentage / 10), 10)
                
                if percentage >= 80:
                    confidence = "🔥 EXCELLENT"
                elif percentage >= 70:
                    confidence = "⭐ VERY HIGH"
                elif percentage >= 60:
                    confidence = "✅ HIGH"
                elif percentage >= 50:
                    confidence = "📊 GOOD"
                else:
                    confidence = "📈 MEDIUM"
                
                print(f"   Photo {photo_num}: {similarity:.4f} ({percentage:.1f}%) {stars}")
                print(f"   Confidence: {confidence}")
            
            # Show statistics
            similarities = [sim for _, sim in photo_similarities]
            best_sim = max(similarities)
            avg_sim = np.mean(similarities)
            min_sim = min(similarities)
            
            print(f"\n📈 Summary for {target_artwork_id}:")
            print(f"   Best similarity: {best_sim:.4f} ({best_sim*100:.1f}%)")
            print(f"   Average similarity: {avg_sim:.4f} ({avg_sim*100:.1f}%)")
            print(f"   Minimum similarity: {min_sim:.4f} ({min_sim*100:.1f}%)")
            print(f"   Consistency: {np.std(similarities):.4f} (lower = more consistent)")
        
        return photo_similarities

def main():
    # Initialize
    extractor = MultiArtworkFeatureExtractor()
    search_engine = MultiPhotoSimilaritySearch()
    
    # Test images
    test_images = [
        "../test_multi/p2 (9).jpg",
        "../test_multi/p1 (13).jpg"
    ]
    
    print("🎯 MULTI-PHOTO ARTWORK SIMILARITY TEST")
    print("=" * 60)
    
    # Test 1: Search across all artworks
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n🔄 Testing with: {os.path.basename(test_image)}")
            
            query_features = extractor.extract_single_features(test_image)
            if query_features is not None:
                results = search_engine.search_across_all_artworks(query_features, top_k=3)
                
                print("🏆 Top matching artworks:")
                for i, result in enumerate(results):
                    artwork_id = result['artwork_id']
                    best_sim = result['best_similarity']
                    avg_sim = result['avg_similarity']
                    photo_count = result['photo_count']
                    
                    print(f"  {i+1}. {artwork_id}")
                    print(f"     Best match: {best_sim:.4f} ({best_sim*100:.1f}%)")
                    print(f"     Avg across {photo_count} photos: {avg_sim:.4f} ({avg_sim*100:.1f}%)")
    
    # Test 2: Detailed comparisons
    print("\n" + "=" * 60)
    print("📋 DETAILED COMPARISONS")
    
    # Test p2 (9).jpg with p2 artwork
    test_image1 = "../test_multi/p2 (9).jpg"
    if os.path.exists(test_image1):
        print(f"\n🧪 Testing {os.path.basename(test_image1)} with p2 artwork:")
        search_engine.detailed_artwork_comparison(test_image1, extractor, "p2")
    
    # Test p1 (13).jpg with p1 artwork  
    test_image2 = "../test_multi/p1 (13).jpg"
    if os.path.exists(test_image2):
        print(f"\n🧪 Testing {os.path.basename(test_image2)} with p1 artwork:")
        search_engine.detailed_artwork_comparison(test_image2, extractor, "p1")
    
    # Cross-test: p2 image with p1 artwork
    if os.path.exists(test_image1):
        print(f"\n🧪 Cross-test: {os.path.basename(test_image1)} with p1 artwork:")
        search_engine.detailed_artwork_comparison(test_image1, extractor, "p1")
    
    # Cross-test: p1 image with p2 artwork
    if os.path.exists(test_image2):
        print(f"\n🧪 Cross-test: {os.path.basename(test_image2)} with p2 artwork:")
        search_engine.detailed_artwork_comparison(test_image2, extractor, "p2")

if __name__ == "__main__":
    main()