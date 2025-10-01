import os
import numpy as np
import json
from feature_extractor import ArtworkFeatureExtractor

class OptimizedSimilaritySearch:
    def __init__(self, storage_dir="../artwork_features"):
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self.features_dir = os.path.join(storage_dir, "features")
        
        # Load metadata
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Load all features
        self.feature_matrix = []
        self.artwork_ids = []
        
        for artwork_id, meta in self.metadata.items():
            feature_path = meta['feature_path']
            features = np.load(feature_path)
            self.feature_matrix.append(features)
            self.artwork_ids.append(artwork_id)
        
        self.feature_matrix = np.array(self.feature_matrix)
        
        # OPTIMAL: Simple L2 normalization only
        norms = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        self.feature_matrix = self.feature_matrix / norms
        
        print(f"✅ Loaded {len(self.artwork_ids)} artworks with optimal preprocessing")

    def power_cosine_similarity(self, vec1, vec2, power=1.5):
        """Cosine similarity with power scaling to emphasize high similarities"""
        cosine_sim = np.dot(vec1, vec2)
        # Apply power to stretch high values and compress low values
        return cosine_sim ** power

    def enhanced_search(self, query_features, top_k=5, power=1.5):
        """Enhanced search with power scaling"""
        # Normalize query
        query_norm = query_features / np.linalg.norm(query_features)
        
        similarities = []
        
        for i, stored_features in enumerate(self.feature_matrix):
            sim = self.power_cosine_similarity(query_norm, stored_features, power)
            similarities.append((self.artwork_ids[i], sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

    def search_by_image_optimized(self, image_path, extractor, top_k=5):
        """Optimized search with power scaling"""
        print(f"🔍 Processing: {os.path.basename(image_path)}")
        query_features = extractor.extract_single_features(image_path)
        if query_features is None:
            return None
        
        # Use power=1.3 to boost high similarities
        results = self.enhanced_search(query_features, top_k, power=1.3)
        return results

def main():
    # Initialize
    extractor = ArtworkFeatureExtractor()
    optimized_search = OptimizedSimilaritySearch()
    
    query_images = [
        "../query_artworks/query_image1.jpg",
        "../query_artworks/query_image2.jpg", 
        "../query_artworks/query_image3.jpg",
        "../query_artworks/query_image4.jpg",
        "../query_artworks/query_image5.jpg", 
        "../query_artworks/query_image6.jpg",
        "../query_artworks/query_image7.jpg"
    ]
    
    print("OPTIMIZED SIMILARITY SEARCH")
    print("=" * 50)
    
    # Test different power values to find optimal
    test_powers = [1.0, 1.3, 1.5, 2.0]
    
    for query_image in query_images:
        if os.path.exists(query_image):
            print(f"\n🎨 Query: {os.path.basename(query_image)}")
            query_features = extractor.extract_single_features(query_image)
            
            if query_features is not None:
                best_results = None
                best_power = 1.0
                best_score = 0
                
                # Find best power parameter for this query
                for power in test_powers:
                    results = optimized_search.enhanced_search(query_features, top_k=1, power=power)
                    if results and results[0][1] > best_score:
                        best_score = results[0][1]
                        best_results = results
                        best_power = power
                
                # Get full results with best power
                full_results = optimized_search.enhanced_search(query_features, top_k=5, power=best_power)
                
                print(f"   Optimal power: {best_power}")
                print("   Top matches:")
                
                for i, (artwork_id, similarity) in enumerate(full_results):
                    original_path = optimized_search.metadata[artwork_id]['image_path']
                    
                    # Enhanced visualization
                    percentage = similarity * 100
                    stars = "★" * min(int(percentage / 10), 10)
                    
                    if percentage >= 80:
                        confidence = "🔥 EXCELLENT"
                        color = "🟢"
                    elif percentage >= 70:
                        confidence = "⭐ VERY HIGH" 
                        color = "🟡"
                    elif percentage >= 60:
                        confidence = "✅ HIGH"
                        color = "🟠"
                    elif percentage >= 50:
                        confidence = "📊 GOOD"
                        color = "🔵"
                    else:
                        confidence = "📈 MEDIUM"
                        color = "🔴"
                    
                    print(f"     {i+1}. {artwork_id} {color}")
                    print(f"        Score: {similarity:.4f} ({percentage:.1f}%) {stars}")
                    print(f"        Confidence: {confidence}")
                    
                    # Highlight perfect matches
                    if artwork_id == os.path.basename(query_image).replace('query_', '').replace('.jpg', ''):
                        print(f"        🎯 PERFECT MATCH!")
            else:
                print("   ❌ Failed to extract features")
        else:
            print(f"❌ Image not found: {query_image}")
    
    print("\n" + "=" * 50)
    print("Similarity search completed")

if __name__ == "__main__":
    main()