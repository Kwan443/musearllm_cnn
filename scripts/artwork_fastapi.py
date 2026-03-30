import os
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from PIL import Image
import io
import torch
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re

from multi_feature_extractor import MultiArtworkFeatureExtractor
from multi_similarity_search import MultiPhotoSimilaritySearch

# Initialize FastAPI app
app = FastAPI(
    title="Artwork Recognition API",
    description="API for recognizing artworks using CNN features and similarity search",
    version="1.0.0"
)

filterList = ['ebay', 'https://www.amazon.com', 'https://www.youtube.com', 'etsy', 'https://www.facebook.com', 'https://us.amazon.com', ' https://www.walmart.com']

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the feature extractor and search engine
extractor = None
search_engine = None

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    message: str
    artworks_loaded: int

class ArtworkInfo(BaseModel):
    artwork_id: str
    photo_count: int
    photo_paths: List[str]

class ArtworksResponse(BaseModel):
    artworks: List[ArtworkInfo]
    total_artworks: int

class GoogleArtWorkResponse(BaseModel):
    success: bool
    response: dict

class MatchResult(BaseModel):
    rank: int
    artwork_id: str
    similarity_score: float
    similarity_percentage: float
    confidence: str
    average_similarity: float
    photos_compared: int

class RecognitionResponse(BaseModel):
    success: bool
    query_image: str
    matches: List[MatchResult]

class URLRequest(BaseModel):
    image_url: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class CNNInfoResponse(BaseModel):
    modelName: Optional[str] = None
    modelPath: Optional[str] = None
    artworks_loaded: Optional[int] = 0
    feature_files: Optional[int] = 0

def extract_single_features_from_image(extractor, image):
    """Extract features from PIL Image object"""
    try:
        inputs = extractor.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = extractor.model(**inputs)
        
        if hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output.squeeze().cpu().numpy()
        else:
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
        return features
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize models when the application starts"""
    global extractor, search_engine
    
    print("🔄 Initializing models...")
    try:
        extractor = MultiArtworkFeatureExtractor()
        search_engine = MultiPhotoSimilaritySearch()
        print("✅ Models initialized successfully!")
    except Exception as e:
        # Don't raise here — allow the API to start even if no features/photos exist.
        print(f"❌ Failed to initialize models: {str(e)}")
        print("⚠️ Continuing startup with partial initialization. Some endpoints may be limited until models/index are available.")
        # If extractor partially initialized, keep it; otherwise set to None
        try:
            if 'extractor' in locals() and extractor:
                pass
            else:
                extractor = None
        except Exception:
            extractor = None

        try:
            search_engine = None
        except Exception:
            search_engine = None

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirects to docs"""
    return {"message": "Artwork Recognition API - Visit /docs for API documentation"}

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and get basic information"
)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Artwork Recognition API is running",
        artworks_loaded=len(search_engine.artwork_ids) if search_engine else 0
    )

@app.get(
    "/artworks",
    response_model=ArtworksResponse,
    summary="Get Artworks",
    description="Get list of all available artworks in the database"
)
async def get_artworks():
    """Get list of all available artworks"""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    artworks = []
    for artwork_id, meta in search_engine.metadata.items():
        artworks.append(ArtworkInfo(
            artwork_id=artwork_id,
            photo_count=meta['photo_count'],
            photo_paths=[os.path.basename(path) for path in meta['photo_paths']]
        ))
    
    return ArtworksResponse(
        artworks=artworks,
        total_artworks=len(artworks)
    )

@app.get(
    "/artworkByGoogle",
    response_model=GoogleArtWorkResponse,
    summary="Get the artwork from google",
    description="Get information of an artwork by Google lens"
)
async def getArtworkByGoogle(imageUrl: str = Query(..., description="Image URL to analyze")):
    """Get information of an artwork by Google lens"""
    print("Image URL:", imageUrl)
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    load_dotenv()
    SEPRAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    print(SEPRAPI_API_KEY)
    params={
        "api_key": SEPRAPI_API_KEY,
        "engine": "google_lens",
        "url": imageUrl, # Image URL
    }

    search = requests.get("https://serpapi.com/search", params=params)
    response = search.json()
    
    # Regular expression pattern — matches if "ebay" or "amazon" appears
    pattern = re.compile(r'(ebay|amazon)', re.IGNORECASE)
    topResults = response.get("visual_matches")

    for result in topResults:
        for filterWord in filterList:
            if re.search(filterWord,result.get("link")):
                print(f"🔍 Skipping URL due to filter match: {result.get('link')}")
                topResults.remove(result)
                break
    top3ResultUrls = [result.get("link") for result in topResults[:3]]
    top3ResultsResponse = {}
    
    for resultURL in top3ResultUrls:
        responseObj = {}
        if not resultURL:
            raise HTTPException(status_code=404, detail="First match has no link.")
        print("🔗 Scraping URL:", resultURL)

        # Step 3 — Scrape webpage content
        page_response = requests.get(resultURL, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        print("Fetch status:", page_response.status_code)

        if page_response.status_code == 200:
        
            # Step 4 — Parse content with BeautifulSoup
            soup = BeautifulSoup(page_response.text, "html.parser")

            # Example extraction: title and paragraph text
            page_title = soup.title.string if soup.title else "No title"
            responseObj["match_title"] = page_title
            match_paragraph = []
            for i, p_tag in enumerate(soup.find_all("p")):
                if i >= 3:
                    break
                text = p_tag.get_text(strip=True)
                if text:
                    match_paragraph.append(text)
            responseObj["match_paragraph"] = match_paragraph
            # Step 5 — Extract first 3 tables and all their cells
            match_table = []
            for i, t in enumerate(soup.find_all("table")):
                if i >= 2:
                    break
                # Extract cells from all rows
                rows = []
                for tr in t.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    if cells:
                        rows.append(cells)

                # Convert this table (list of rows) into a single readable string
                if rows:
                    # " | " separates cells, "\n" separates rows
                    table_text = "\n".join(" | ".join(row) for row in rows)
                    match_table.append(table_text)
            responseObj["match_table"] = match_table
            top3ResultsResponse[page_title] = {"match_paragraph":match_paragraph,"match_table":match_table}
            print()
    return GoogleArtWorkResponse(
        success=True,
        response=top3ResultsResponse
    )

@app.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Recognize Artwork",
    description="Upload an image and get the top 3 most similar artworks with similarity scores",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def recognize_artwork(image: UploadFile = File(...)):
    """
    Recognize artwork from uploaded image and return top 3 matches
    
    - **image**: Image file (JPG, JPEG, PNG)
    """
    try:
        # Check if models are initialized
        if not extractor or not search_engine:
            raise HTTPException(status_code=500, detail="Models not initialized")
        
        # Check file type
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        file_extension = image.filename.split('.')[-1].lower() if '.' in image.filename else ''
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload JPG, JPEG, or PNG"
            )
        
        # Read and process the image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Extract features from the uploaded image
        print(f"🔍 Processing uploaded image: {image.filename}")
        query_features = extract_single_features_from_image(extractor, pil_image)
        
        if query_features is None:
            raise HTTPException(
                status_code=500, 
                detail="Failed to extract features from image"
            )
        
        # Search for similar artworks
        results = search_engine.search_across_all_artworks(query_features, top_k=3)
        
        # Format the response
        matches = []
        for i, result in enumerate(results):
            artwork_id = result['artwork_id']
            best_similarity = result['best_similarity']
            avg_similarity = result['avg_similarity']
            photo_count = result['photo_count']
            
            # Determine confidence level
            percentage = best_similarity * 100
            if percentage >= 80:
                confidence = "VERY_HIGH"
            elif percentage >= 70:
                confidence = "HIGH"
            elif percentage >= 60:
                confidence = "MEDIUM_HIGH"
            elif percentage >= 50:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            match_result = MatchResult(
                rank=i + 1,
                artwork_id=artwork_id,
                similarity_score=round(best_similarity, 4),
                similarity_percentage=round(percentage, 2),
                confidence=confidence,
                average_similarity=round(avg_similarity, 4),
                photos_compared=photo_count
            )
            matches.append(match_result)
        
        print(f"✅ Recognition completed for {image.filename}")
        return RecognitionResponse(
            success=True,
            query_image=image.filename,
            matches=matches
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in recognition: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/cnn/info",
    response_model=CNNInfoResponse,
    summary="Get CNN info",
    description="Returns basic info about the feature extractor and stored features"
)
async def get_cnn_info():
    if not extractor:
        raise HTTPException(status_code=500, detail="Extractor not initialized")

    # Count feature files
    features_dir = getattr(extractor, 'features_dir', '../artwork_features_multi/features')
    count = 0
    try:
        if os.path.exists(features_dir):
            count = len([f for f in os.listdir(features_dir) if f.endswith('.npy')])
    except Exception:
        count = 0

    return CNNInfoResponse(
        modelName=getattr(extractor, 'model_path', None),
        modelPath=getattr(extractor, 'model_path', None),
        artworks_loaded=len(search_engine.artwork_ids) if search_engine else 0,
        feature_files=count
    )


def _do_reindex(photos_dir="../photos"):
    global extractor, search_engine
    try:
        # re-run feature extraction over photos dir
        extractor.process_multi_photo_directory(photos_dir)
        # reload search engine so metadata is fresh
        search_engine = MultiPhotoSimilaritySearch()
        print("✅ Reindex completed")
    except Exception as e:
        print(f"❌ Reindex failed: {str(e)}")


@app.post(
    "/cnn/reindex",
    summary="Start full reindex",
    description="Triggers a full reindex (runs in background)"
)
async def reindex_cnn(background_tasks: BackgroundTasks):
    background_tasks.add_task(_do_reindex)
    return JSONResponse(status_code=202, content={"status": "reindex started"})


@app.post(
    "/cnn/extract",
    summary="Extract features for an artwork id",
    description="Trigger feature extraction for a given artwork id (looks for photos in ../photos)"
)
async def extract_for_id(artwork_id: str = Query(..., description="Artwork id to extract")):
    global extractor, search_engine
    if not extractor:
        raise HTTPException(status_code=500, detail="Extractor not initialized")

    photos_dir = "../photos"
    grouped = extractor.group_images_by_prefix(photos_dir)
    if artwork_id not in grouped:
        raise HTTPException(status_code=404, detail=f"No photos found for id {artwork_id}")

    success = extractor.save_multi_photo_features(grouped[artwork_id], artwork_id)
    if not success:
        raise HTTPException(status_code=500, detail="Feature extraction failed")

    # reload search engine
    search_engine = MultiPhotoSimilaritySearch()
    return JSONResponse(status_code=200, content={"status": "extracted", "artwork_id": artwork_id})


@app.post(
    "/photos/upload",
    summary="Upload photo for artwork",
    description="Upload a photo file associated with an artwork id. Saves file to ../photos and triggers feature extraction."
)
async def upload_photo(artwork_id: str = Query(..., description="Artwork id"), file: UploadFile = File(...)):
    """Save an uploaded photo for an artwork and run extraction for that id."""
    global extractor, search_engine
    photos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'photos'))
    os.makedirs(photos_dir, exist_ok=True)

    # Construct a filename that groups by artwork id as prefix
    safe_name = os.path.basename(file.filename)
    save_name = f"{artwork_id} {safe_name}"
    save_path = os.path.join(photos_dir, save_name)

    try:
        contents = await file.read()
        with open(save_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        print(f"❌ Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

    # If extractor is available, run feature extraction for this artwork
    try:
        if extractor:
            extractor.save_multi_photo_features([save_path], artwork_id)
        # Reload search engine to include new features if possible
        try:
            search_engine = MultiPhotoSimilaritySearch()
        except Exception as e:
            print(f"⚠️ Failed to reload search engine after upload: {e}")
    except Exception as e:
        print(f"❌ Error during extraction/save: {e}")
        # return success for upload but indicate extraction problem
        return JSONResponse(status_code=201, content={"status": "uploaded", "extraction": "failed", "detail": str(e)})

    return JSONResponse(status_code=201, content={"status": "uploaded", "path": save_name})

@app.post(
    "/recognize_url",
    summary="Recognize from URL",
    description="Recognize artwork from image URL (Not implemented yet)",
    responses={501: {"model": ErrorResponse}}
)
async def recognize_artwork_from_url(request: URLRequest):
    """
    Recognize artwork from image URL
    
    - **image_url**: URL of the image to analyze
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": "URL recognition not implemented yet",
            "detail": "Please use the file upload endpoint (/recognize)",
            "image_url": request.image_url
        }
    )

if __name__ == '__main__':
    print("🚀 Starting Artwork Recognition FastAPI...")
    print("📚 Auto-generated docs available at: http://localhost:8000/docs")
    print("🔗 Alternative docs at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "artwork_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload during development
        log_level="info"
    )