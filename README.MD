# Word Agent API Server

This is a FastAPI server that provides a "Word Agent" service. It generates rich, culturally context-aware descriptions and images for words to help with vocabulary learning.

## Prerequisites

- Python 3.8+
- Supabase Account (URL and Key)
- Google Gemini API Key

## Setup & Installation

Follow these steps to set up the project on Windows:

1.  **Create a Virtual Environment**:
    ```powershell
    python -m venv .venv
    ```

2.  **Activate the Virtual Environment**:
    ```powershell
    .\.venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Environment Variables**:
    Ensure you have a `.env` file in the root directory with the following keys:
    ```env
    GEMINI_API_KEY=your_gemini_api_key
    SUPABASE_URL=your_supabase_url
    SUPABASE_KEY=your_supabase_key
    ```

## Running the Server

Run the server using `uvicorn`:

```powershell
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

## API Documentation & Usage

### Interactive Docs (Swagger UI)
You can view the interactive API documentation here:
- **URL**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Generate Word Content

**Endpoint:** `GET /`

**Parameters:**
- `words`: (Required) Word(s) to process. Can be a single word or comma-separated list.
- `lang`: (Optional) Target language.
- `age`: (Optional) User age.
- `level`: (Optional) Proficiency level.
- `style`: (Optional) Image style.
- `country`: (Optional) User country context.

#### Examples:

**1. Single Word Request:**
```
http://127.0.0.1:8000/?words=apple&lang=Arabic&age=25&level=Intermediate&style=Photorealistic&country=Egypt
```

**2. Multiple Words Request (Processed Concurrently):**
```
http://127.0.0.1:8000/?words=cat,dog,bird&lang=Spanish&age=10&level=Beginner&style=Cartoon&country=Spain
```

**Response:**
Returns a JSON object (or list of objects) containing validity status, description, examples, and a generated image URL.

## Docker Support

Build and run the container:

```bash
# Build
docker build -t word-agent .

# Run
docker run -p 8000:8000 --env-file .env word-agent
```

## Project Structure

- `main.py`: FastAPI server entry point.
- `GeminiWrapper.py`: Wrapper for Google GenAI SDK.
- `prompts.py`: Prompt management for the agent.
- `models.py`: Pydantic models for data validation.
- `config.py`: Configuration settings.
- `logger_config.py`: Logging configuration.
- `Dockerfile`: Docker build instructions.

