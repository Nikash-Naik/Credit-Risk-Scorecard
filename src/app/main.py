import pathlib
import sys

# When run directly (python src/app/main.py), src/ is not on sys.path.
# Insert it before any "from app." imports so they resolve correctly.
_src = str(pathlib.Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from fastapi import FastAPI

from app.auth.router import router as auth_router
from app.db import Base, engine
from app.monitor.router import router as monitor_router
from app.predict.router import router as predict_router

app = FastAPI(title="Credit Risk Scorecard API")

Base.metadata.create_all(bind=engine)

app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(monitor_router)


@app.get("/")
def root():
    return {"message": "Credit Risk Scorecard API. See /docs for usage."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)