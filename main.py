"""Entry point. `uvicorn main:app --port 8010`, or `python main.py` (no reload: the models load once)."""

import uvicorn

from app.bootstrap import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8010)
