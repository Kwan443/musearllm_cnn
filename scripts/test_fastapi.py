import requests
import json
import time

def print_response(response, endpoint_name):
    """Print formatted response"""
    print(f"\n{'='*50}")
    print(f"🔍 {endpoint_name}")
    print(f"{'='*50}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_health():
    """Test health endpoint"""
    response = requests.get('http://localhost:8000/health')
    print_response(response, "Health Check")

def test_artworks():
    """Test artworks endpoint"""
    response = requests.get('http://localhost:8000/artworks')
    print_response(response, "Available Artworks")

def test_recognition(image_path):
    """Test artwork recognition with an image file"""
    print(f"\n🖼️ Testing recognition with: {image_path}")
    
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        response = requests.post('http://localhost:8000/recognize', files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Recognition Results:")
        print(f"Query Image: {result['query_image']}")
        print("\n🏆 Top 3 Matches:")
        
        for match in result['matches']:
            print(f"  {match['rank']}. {match['artwork_id']}")
            print(f"     Similarity: {match['similarity_percentage']}%")
            print(f"     Confidence: {match['confidence']}")
            print(f"     Photos Compared: {match['photos_compared']}")
            print(f"     Score: {match['similarity_score']:.4f}")
            print()
    else:
        print(f"❌ Error: {response.json()}")

def test_invalid_file():
    """Test with invalid file type"""
    print("\n🧪 Testing with invalid file type...")
    
    # Create a dummy text file
    files = {'image': ('test.txt', 'this is not an image', 'text/plain')}
    response = requests.post('http://localhost:8000/recognize', files=files)
    
    print_response(response, "Invalid File Test")

def main():
    print("🚀 FastAPI Artwork Recognition Test Client")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    
    # Test basic endpoints
    test_health()
    test_artworks()
    
    # Test with your specified images
    test_images = [
        "../test_multi/p2 (9).jpg",
        "../test_multi/p1 (13).jpg"
    ]
    
    for image_path in test_images:
        test_recognition(image_path)
    
    # Test error case
    test_invalid_file()

if __name__ == '__main__':
    main()