import asyncio
import websockets
import json  # Import JSON for message parsing

connected_clients = set()

async def handle_client(websocket, path):
    """Handle incoming WebSocket connections."""
    print(f"New client connected: {websocket.remote_address} on path: {path}")
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            try:
                # Parse the incoming message as JSON
                data = json.loads(message)
                
                # Validate the message format
                if not isinstance(data, dict) or 'cameraId' not in data or 'streamUrl' not in data:
                    raise ValueError("Invalid message format. Expected a JSON object with 'cameraId' and 'streamUrl'.")

                # Broadcast the message to all connected clients
                for client in connected_clients:
                    if client != websocket:
                        await client.send(json.dumps(data))
            except json.JSONDecodeError:
                print("Error: Received invalid JSON message.")
                await websocket.send(json.dumps({"error": "Invalid JSON format"}))
            except ValueError as e:
                print(f"Error: {e}")
                await websocket.send(json.dumps({"error": str(e)}))
    except websockets.ConnectionClosed as e:
        print(f"Client disconnected: {websocket.remote_address}, Reason: {e}")
    except Exception as e:
        print(f"Unexpected error in WebSocket handler: {e}")
        # Close the connection with an error code
        await websocket.close(code=1011, reason="Internal server error")
    finally:
        connected_clients.remove(websocket)

async def main():
    """Start the WebSocket server."""
    print("Starting WebSocket server on ws://localhost:8765/")
    try:
        async with websockets.serve(handle_client, "localhost", 8765):
            await asyncio.Future()  # Run forever
    except Exception as e:
        print(f"Error starting WebSocket server: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWebSocket server stopped.")
