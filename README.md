# AI Analytics Copilot

An intelligent, modular Analytics Copilot powered by LLMs and Streamlit.

## Project Structure & Responsibilities

This project is built using a clean, scalable modular architecture. Business logic is separated into distinct packages inside the `src/` directory.

- **`app.py`**: The main entry point for the Streamlit frontend. Responsible only for UI layout and routing user interactions to the appropriate backend modules.
- **`src/analysis/`**: Contains core data processing and analytics logic. Handles data transformation, statistical analysis, and aggregating results.
- **`src/llm/`**: Manages all interactions with Large Language Models. Responsible for prompt construction, API client wrappers, response parsing, and error handling for LLM providers.
- **`src/visuals/`**: Responsible for creating charts, graphs, and visual dashboards. Takes processed data and returns visual components that the Streamlit frontend can render.
- **`src/metadata/`**: Manages metadata related to datasets, schemas, and user configurations. Keeps track of what data is available and its structure.
- **`src/context/`**: Handles the state and context of user sessions. Manages chat history, ongoing tasks, and short-term memory required to provide a cohesive conversational experience.
- **`src/utils/`**: Contains cross-cutting concerns like environment variable configuration (`config.py`) and centralized logging (`logger.py`).

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and fill in your API keys and configuration.
   ```bash
   cp .env.example .env
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

This app runs as-is on [Streamlit Community Cloud](https://share.streamlit.io) (free). It is **not** compatible with Vercel/serverless hosts, which can't run Streamlit's long-running websocket server.

1. **Rotate your API key first.** `.env` was previously committed, so treat the old key as exposed — generate a new `GEMINI_API_KEY` in Google AI Studio.
2. **Push to GitHub.** `.env` and `.streamlit/secrets.toml` are gitignored, so no secrets are published.
3. **Create the app** at share.streamlit.io → "New app" → pick this repo, branch, and `app.py`.
4. **Add secrets** (app → Settings → Secrets), using the format in `.streamlit/secrets.toml.example`:
   ```toml
   GEMINI_API_KEY = "your-new-key"
   APP_PASSWORD   = "a-shared-password"   # testers must enter this to use the app
   ```
   If `APP_PASSWORD` is omitted the app is open to anyone with the link (and uses your key), so set it for a shared preview.
5. **Share the link.** Note that uploaded files and learned rules are stored on the app's ephemeral disk — they reset on restart and are shared across testers (this is a single-process preview, not a multi-user backend).
