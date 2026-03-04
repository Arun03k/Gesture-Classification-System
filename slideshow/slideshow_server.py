import pathlib
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

BASE_DIR = pathlib.Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Reveal.js Slideshow Controller")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    index_file = STATIC_DIR / "index.html"
    return index_file.read_text(encoding="utf-8")


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
        to_remove: List[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        manager.disconnect(websocket)


class Event(BaseModel):
    command: str


@app.post("/event")
async def send_event(event: Event) -> dict:
    await manager.broadcast(event.command)
    return {"status": "ok", "command": event.command}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8800)
