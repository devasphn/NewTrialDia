# Real-Time Dialogue Agent with Nari Labs Dia TTS

This project provides a complete, self-contained AI agent that interacts in real-time through voice via a web interface. It uses a state-of-the-art model pipeline to generate truly natural, expressive, and human-like dialogue, including non-verbal sounds like laughter.

This is not a standard text-to-speech agent; it is a **dialogue generation system**.

## Core Technology Pipeline
*   **Web UI:** Gradio for a fast, interactive web interface.
*   **Speech-to-Text (STT):** `faster-whisper` (distil-large-v3 model) for fast and accurate transcription.
*   **Language Model (LLM):** Meta's `Llama-3-8B-Instruct` (in 4-bit), prompted to generate dialogue scripts.
*   **Text-to-Dialogue (TTD):** **Nari Labs Dia** (`nari-labs/Dia-1.6B-0626`) for high-fidelity, expressive dialogue synthesis.

## Prerequisites
*   **Hardware:** A machine with a modern NVIDIA GPU with at least 16GB of VRAM (e.g., A4500, A6000, RTX 3090, RTX 4090).
*   **Operating System:** Linux (recommended for Runpod).
*   **Software:**
    *   Python 3.10+
    *   `git` and `git-lfs` (for downloading models).
    *   `wget`

## Setup and Installation

**Follow these steps precisely.**

1.  **Clone the Repository**
    ```bash
    # Replace with your repository URL
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Download Reference Audio & Create Transcript**
    Dia requires a reference audio file and its exact transcript to clone a voice.
    
    First, download the audio:
    ```bash
    wget -O speaker.wav https://huggingface.co/datasets/mca-ark/libri-tts-test-clean/resolve/main/116_288045_000020_000001.wav
    ```
    Next, create a text file containing the exact words spoken in that audio:
    ```bash
    echo "[S1] and that he was not authorized to use it in any way." > speaker_transcript.txt
    ```

4.  **Install Bleeding-Edge Transformers**
    Dia's integration is very new and requires the main branch of `transformers`.
    ```bash
    pip install git+https://github.com/huggingface/transformers.git
    ```

5.  **Install All Other Dependencies**
    Now, install the remaining libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

6.  **Log in to Hugging Face (Required for Llama 3)**
    Log in to your Hugging Face account to download the Llama 3 model.
    ```bash
    huggingface-cli login
    # Paste your Hugging Face token when prompted
    ```

## Running the Agent

1.  **Start the Web Server**
    Run the application using the `app.py` script. The first run will download several GB of models and may take a long time.
    ```bash
    python app.py
    ```

2.  **Access the UI**
    The script will start a web server on port 7860. On Runpod, a button will appear to connect.

3.  **Interact with the Agent**
    *   Click "Tap to Talk" and speak.
    *   The agent will generate a response in dialogue format and speak it.

4.  **Stopping the Agent**
    To stop the server, press `Ctrl+C` in the terminal.
