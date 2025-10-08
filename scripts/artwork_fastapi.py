import os
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from PIL import Image
import io
import torch

from multi_feature_extractor import MultiArtworkFeatureExtractor
from multi_similarity_search import MultiPhotoSimilaritySearch

# Initialize FastAPI app
app = FastAPI(
    title="Artwork Recognition API",
    description="API for recognizing artworks using CNN features and similarity search",
    version="1.0.0"
)

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
        print(f"❌ Failed to initialize models: {str(e)}")
        raise e

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