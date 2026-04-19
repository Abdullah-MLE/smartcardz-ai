from main import gemini_client

print(f"Client API type: {'Vertex' if gemini_client._api_client.vertexai else 'AI Studio'}")

try:
    print("\n--- Available Image Models ---")
    for m in gemini_client.models.list():
        name = m.name.lower() if m.name else ""
        display = m.display_name.lower() if m.display_name else ""
        if "image" in name or "image" in display or "imagen" in name:
            print(f"Name: {m.name} | Display Name: {m.display_name}")
except Exception as e:
    print(f"Error fetching models: {e}")
