from __future__ import annotations
from fastapi import WebSocket


class WebSocketManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)
        await websocket.close()

    async def close_all(self) -> None:
        for connection in self.active_connections.copy():
            await self.disconnect(connection)
