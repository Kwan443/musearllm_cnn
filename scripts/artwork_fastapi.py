import os
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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

class GoogleArtWorkResponse(BaseModel):
    success: bool
    match_title: str
    match_paragraph: List[str]
    match_table: List[str]

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
    

    match_paragraph = []
    count = -1
    while match_paragraph == []:
        count+=1
        firstResult = response.get("visual_matches")[count]
        firstResultUrl = firstResult.get("link")

        if not firstResultUrl:
            raise HTTPException(status_code=404, detail="First match has no link.")
        print("🔗 Scraping URL:", firstResultUrl)

        # Step 3 — Scrape webpage content
        page_response = requests.get(firstResultUrl, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

        if page_response.status_code != 200:
            raise HTTPException(status_code=page_response.status_code, detail="Failed to fetch webpage")

        # Step 4 — Parse content with BeautifulSoup
        soup = BeautifulSoup(page_response.text, "html.parser")

        # Example extraction: title and paragraph text
        page_title = soup.title.string if soup.title else "No title"
        for i, p_tag in enumerate(soup.find_all("p")):
            if i >= 3:
                break
            text = p_tag.get_text(strip=True)
            if text:
                match_paragraph.append(text)

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
        # print("Content: ", {match_paragraph,match_table})
        return GoogleArtWorkResponse(
            success=True,
            match_title=page_title,
            match_paragraph = match_paragraph or [],
            match_table=match_table or []
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