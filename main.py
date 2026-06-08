import uvicorn

if __name__ == "__main__":
    from src import app
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8000
    )
