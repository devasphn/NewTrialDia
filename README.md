# Real-Time Speech-to-Speech AI Agent (Dia TTS Version)

This project provides a complete, self-contained AI agent that interacts in real-time through voice via a web interface. It uses a state-of-the-art model pipeline for a fluid and emotionally aware conversational experience.

This version has been upgraded to use the advanced **Dia** model for dialogue generation, offering more expressive and natural-sounding speech.

## Core Technology Pipeline
*   **Web UI:** Gradio for a fast, interactive web interface.
*   **Speech-to-Text (STT):** `faster-whisper` (distil-large-v3 model) for fast and accurate transcription.
*   **Language Model (LLM):** Meta's `Llama-3-8B-Instruct` (in 4-bit) for intelligent response generation.
*   **Text-to-Speech (TTS):** Nari Labs' `Dia-1.6B-0626` for high-quality, expressive dialogue synthesis.

## Prerequisites
*   **Hardware:** A machine with a modern NVIDIA GPU with at least 16GB of VRAM (e.g., A4500, A6000, RTX 3090, RTX 4090).
*   **Operating System:** Linux (recommended for Runpod).
*   **Software:**
    *   Python 3.10+
    *   `git` and `git-lfs`

## Setup and Installation on Runpod

Follow these steps precisely.

1.  **Clone Your New Repository**
    Create a new repository on GitHub called `NewTrialDia` and clone it to your Runpod instance.
    ```bash
    # Replace with your repository URL
    git clone https://github.com/YOUR_USERNAME/NewTrialDia.git
    cd NewTrialDia
    ```

2.  **Create and Activate Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install System Dependencies**
    The Dia model may require some additional system libraries.
    ```bash
    sudo apt-get update
    sudo apt-get install -y libsndfile1 ffmpeg
    ```

4.  **Install Python Dependencies**
    This is a two-step process. First, install the main requirements. Then, install the specific version of `transformers` needed for Dia.
    ```bash
    pip install -r requirements.txt
    pip install git+https://github.com/huggingface/transformers.git
    ```

5.  **Log in to Hugging Face**
    This is required for Llama 3.
    ```bash
    huggingface-cli login
    # Paste your token when prompted
    ```

## Running the Agent

1.  **Start the Web Server**
    ```bash
    python app.py
    ```
    The first time you run this, it will download all necessary models (Whisper, Llama 3, Dia, and Descript Audio Codec). This is a large download (25-30 GB) and will take a significant amount of time.

2.  **Access the UI**
    The script will start a web server. On Runpod, a button will appear to "Connect to Port 7860" (or whichever port you configure). Click it to open the UI in your browser.

3.  **Interact with the Agent**
    *   Tap the microphone and speak.
    *   Your transcription will appear, followed by the agent's text response.
    *   An audio player will appear with the agent's spoken response, which will play automatically.

4.  **Stopping the Agent**
    Press `Ctrl+C` in the terminal.
