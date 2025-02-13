from fastapi import FastAPI,HTTPException
from backend import process_query  # Import only the function now
from pydantic import BaseModel

from base64 import b64decode


app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Hello, World! FastAPI is running."}

@app.post("/process_query")
def query_endpoint(request: QueryRequest):
    """API endpoint to retrieve documents based on the user's query."""
    print(f"Received query: {request.query}")
    try:
        result = process_query(request.query)
        return {"insight": result}
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process query")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
