import asyncio
import websockets

async def test():
    async with websockets.connect('ws://127.0.0.1:8080/ws') as websocket:
        msg = await websocket.recv()
        print("Received:", msg)

asyncio.run(test())
