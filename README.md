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
