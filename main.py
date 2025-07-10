import torch
import transformers
import faster_whisper
import gradio as gr
import numpy as np
import time
import os
import soundfile as sf

# --- Configuration ---
STT_MODEL = "distil-large-v3"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# NEW: Using Nari Labs Dia for dialogue generation
TTS_MODEL = "nari-labs/Dia-1.6B-0626" 
SPEAKER_REFERENCE_WAV = "speaker.wav"
SPEAKER_TRANSCRIPT_FILE = "speaker_transcript.txt"
OUTPUT_WAV_FILE = "output.wav"

class DialogueS2SAgent:
    """
    A real-time Speech-to-Speech agent using Nari Labs Dia for dialogue generation.
    """
    def __init__(self):
        print("--- Initializing Dialogue S2S Agent ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        print(f"Using device: {self.device.upper()} with dtype: {self.torch_dtype}")

        # 1. STT (Speech-to-Text) Model
        print(f"Loading STT model: {STT_MODEL}...")
        self.stt_model = faster_whisper.WhisperModel(
            STT_MODEL, device=self.device, compute_type="float16"
        )
        print("STT model loaded.")

        # 2. LLM (Language Model)
        print(f"Loading LLM: {LLM_MODEL}...")
        self.llm_pipeline = transformers.pipeline(
            "text-generation",
            model=LLM_MODEL,
            model_kwargs={"torch_dtype": self.torch_dtype},
            device_map=self.device,
        )
        print("LLM loaded.")

        # 3. NEW TTD (Text-to-Dialogue) Model - Nari Labs Dia
        print(f"Loading TTD model: {TTS_MODEL}...")
        self.tts_model = transformers.DiaForConditionalGeneration.from_pretrained(
            TTS_MODEL, torch_dtype=self.torch_dtype
        ).to(self.device)
        self.tts_processor = transformers.AutoProcessor.from_pretrained(TTS_MODEL)
        
        # Load the reference audio and its transcript for voice cloning
        if not os.path.exists(SPEAKER_REFERENCE_WAV) or not os.path.exists(SPEAKER_TRANSCRIPT_FILE):
            raise FileNotFoundError(
                f"Ensure both '{SPEAKER_REFERENCE_WAV}' and '{SPEAKER_TRANSCRIPT_FILE}' exist. "
                "Please follow the README setup instructions."
            )
        
        with open(SPEAKER_TRANSCRIPT_FILE, 'r') as f:
            self.reference_transcript = f.read().strip()

        print("TTD model and reference audio loaded.")

        if os.path.exists(OUTPUT_WAV_FILE):
            os.remove(OUTPUT_WAV_FILE)
            
        print("\n--- Agent is Ready ---")

    def transcribe_audio(self, audio_filepath: str) -> str:
        if not audio_filepath: return ""
        print("Transcribing audio...")
        segments, _ = self.stt_model.transcribe(audio_filepath, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        print(f"User (transcribed): {transcription}")
        return transcription

    def generate_dialogue_script(self, user_text: str) -> str:
        # NEW: The system prompt instructs the LLM to create a script for Dia.
        # It asks the LLM to act as a scriptwriter.
        system_prompt = (
            "You are a scriptwriter. Create a short, natural-sounding dialogue. "
            "The user speaks as [S2]. You, as a friendly and helpful AI named Deva, speak as [S1]. "
            "Use conversational language. You can use non-verbal cues like (laughs), (sighs), or (clears throat) to make the dialogue more realistic. "
            "The user's line is already provided. Write only the response for [S1]."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[S2] {user_text}"},
        ]
        
        terminators = [
            self.llm_pipeline.tokenizer.eos_token_id,
            self.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.llm_pipeline(
            messages,
            max_new_tokens=150,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
        )
        
        assistant_response = outputs[0]["generated_text"][-1]['content']
        # Clean up the response to ensure it only contains the [S1] part
        if "[S1]" in assistant_response:
             assistant_response = assistant_response.split("[S1]")[1].strip()
        
        # Re-add the tag for Dia
        assistant_script = f"[S1] {assistant_response}"
        print(f"Agent (script): {assistant_script}")
        return assistant_script, assistant_response # Return both the script for Dia and clean text for chat

    def convert_script_to_speech(self, dialogue_script: str) -> str:
        """
        Converts a dialogue script to speech using Dia with voice cloning.
        """
        print("Synthesizing dialogue...")
        # For voice cloning, Dia's prompt is the reference transcript followed by the new script.
        full_script = self.reference_transcript + "\n" + dialogue_script

        inputs = self.tts_processor(
            text=full_script,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.tts_model.generate(
                **inputs,
                audio_prompt=SPEAKER_REFERENCE_WAV,
                max_new_tokens=3072, # Max audio length
                guidance_scale=3.0,
                temperature=0.2, # Lower temperature for less randomness in voice
                top_k=5
            )
        
        waveform = self.tts_processor.batch_decode(output)[0]["audio"][0]
        sample_rate = self.tts_processor.sampling_rate

        sf.write(OUTPUT_WAV_FILE, waveform, sample_rate)
        return OUTPUT_WAV_FILE

    def process_conversation_turn(self, audio_filepath: str, chat_history: list):
        if audio_filepath is None: return chat_history, None
        user_text = self.transcribe_audio(audio_filepath)
        if not user_text.strip(): return chat_history, None
            
        chat_history.append({"role": "user", "content": user_text})

        dialogue_script, clean_response = self.generate_dialogue_script(user_text)
        chat_history.append({"role": "assistant", "content": clean_response})
        
        agent_audio_path = self.convert_script_to_speech(dialogue_script)

        return chat_history, agent_audio_path

def build_ui(agent: DialogueS2SAgent):
    with gr.Blocks(theme=gr.themes.Soft(), title="Dialogue Agent with Dia TTS") as demo:
        gr.Markdown("# Real-Time Dialogue Agent with Nari Labs Dia")
        gr.Markdown("Tap the microphone, speak, and the agent will respond with an expressive, dialogue-generated voice.")
        chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot", height=500, type="messages")
        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Tap to Talk")
            audio_output = gr.Audio(label="Agent Response", autoplay=True, visible=True)
        
        def handle_interaction(audio_filepath, history):
            history = history or []
            return agent.process_conversation_turn(audio_filepath, history)

        mic_input.stop_recording(
            fn=handle_interaction, inputs=[mic_input, chatbot], outputs=[chatbot, audio_output]
        )
        clear_button = gr.Button("Clear Conversation")
        clear_button.click(lambda: ([], None), None, [chatbot, audio_output])
    return demo

if __name__ == "__main__":
    agent = DialogueS2SAgent()
    ui = build_ui(agent)
    ui.launch(server_name="0.0.0.0", server_port=7860)
