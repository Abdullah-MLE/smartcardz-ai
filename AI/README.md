# SmartCardz AI

A simple API to get information about English words.

## Prerequisites

- Python 3.7+

## How to Run

1.  **Create a virtual environment:**
    ```sh
    python -m venv .venv
    ```

2.  **Activate the virtual environment:**
    - On Windows:
      ```sh
      .venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```sh
      source .venv/bin/activate
      ```

3.  **Install dependencies:**
    ```sh
    pip cache purge
    pip install --use-deprecated=legacy-resolver -r requirements.txt

    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    - Rename `.env.example` to `.env`.
    - Add your Google API key to the `.env` file:
      ```
      GOOGLE_API_KEY=your_api_key_here
      ```

5.  **Run the application:**
    ```sh
    uvicorn main:app --host 0.0.0.0 --port 10000 --reload
    ```
    The API will be running at `http://localhost:10000`.