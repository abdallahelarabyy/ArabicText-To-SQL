import os
from typing import Optional
from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import FileResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel  # type: ignore

from rag_system import RAGSystem  # Your RAG logic


# Load Gemini API key
gemini_api_key = os.environ.get(
    "GEMINI_API_KEY", "AIzaSyDBUfMj2MkaW2yGH1sRr_6H5-FFFcYDrCo"
)

# Initialize RAG system
rag_system = RAGSystem(gemini_api_key=gemini_api_key)

# Build or load the index on startup
try:
    rag_system.build_index("AR_spider.jsonl", force_rebuild=False)
except Exception as e:
    print(f"Warning: Could not load or build index: {e}")

# Initialize FastAPI
app = FastAPI(
    title="RAG System API",
    description="A FastAPI-based interface for querying and indexing a RAG system",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Request/Response Models
# ---------------------------


class QueryRequest(BaseModel):
    question: str
    # db_id: Optional[str] = None
    top_k: int = 5


class QueryResponse(BaseModel):
    sql_query: str


class BuildIndexRequest(BaseModel):
    jsonl_path: str = "AR_spider.jsonl"
    force_rebuild: bool = False

# Add these request/response models


# ---------------------------
# Static Files + Endpoints
# ---------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        print(f"Processing query: {request.question}")
        sql_query = rag_system.query(
            question=request.question,
            # db_id=request.db_id,
            top_k=request.top_k
        )
        print(f"Generated SQL: {sql_query}")
        return {"sql_query": sql_query}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this endpoint


# Add this endpoint


@app.post("/build_index")
async def build_index(request: BuildIndexRequest):
    try:
        rag_system.build_index(
            request.jsonl_path,
            force_rebuild=request.force_rebuild
        )
        return {"message": "Index built successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Run Locally
# ---------------------------

if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)


