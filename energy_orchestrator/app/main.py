from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    context = {
        "request": request,
        "current_price": 0.123,
        "message": "Energy Orchestrator UI - stap 1 (skelet)"
    }
    return templates.TemplateResponse("index.html", context)
