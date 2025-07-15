import torch
import transformers
import faster_whisper
import gradio as gr
import numpy as np
import soundfile as sf
import time
import os

# --- Configuration ---
# Models
STT_MODEL = "distil-large-v3"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TTS_MODEL = "nari-labs/Dia-1.6B-0626"
# Output audio file
OUTPUT_WAV_FILE = "output.wav"

class RealTimeS2SAgent:
    """
    A real-time Speech-to-Speech agent integrating the Dia TTS model.
    """
    def __init__(self):
        """
        Initializes all the models and necessary components.
        """
        print("--- Initializing S2S Agent with Dia TTS ---")
        if not torch.cuda.is_available():
            raise RuntimeError("This application requires a GPU to run.")
            
        self.device = "cuda"
        print(f"Using device: {self.device.upper()}")

        # 1. STT (Speech-to-Text) Model
        print(f"Loading STT model: {STT_MODEL}...")
        self.stt_model = faster_whisper.WhisperModel(
            STT_MODEL, 
            device=self.device, 
            compute_type="float16"
        )
        print("STT model loaded.")

        # 2. LLM (Language Model)
        print(f"Loading LLM: {LLM_MODEL}...")
        self.llm_pipeline = transformers.pipeline(
            "text-generation",
            model=LLM_MODEL,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self.device,
        )
        print("LLM loaded.")

        # 3. TTS (Text-to-Speech) Model - Dia by Nari Labs
        print(f"Loading Dia TTS model: {TTS_MODEL}...")
        self.tts_processor = transformers.AutoProcessor.from_pretrained(TTS_MODEL)
        self.tts_model = transformers.DiaForConditionalGeneration.from_pretrained(
            TTS_MODEL, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        print("Dia TTS model loaded.")
        
        if os.path.exists(OUTPUT_WAV_FILE):
            os.remove(OUTPUT_WAV_FILE)
            
        print("\n--- Agent is Ready ---")

    def transcribe_audio(self, audio_filepath: str) -> str:
        """Transcribes audio to text."""
        if not audio_filepath: return ""
        print("Transcribing audio...")
        segments, _ = self.stt_model.transcribe(audio_filepath, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        print(f"User: {transcription}")
        return transcription

    def generate_response(self, chat_history: list) -> str:
        """Generates a response from the LLM."""
        messages = [
            {"role": "system", "content": "You are a friendly and helpful conversational AI. Your name is Deva. Keep your responses concise, conversational, and expressive. Use laughs or sighs where appropriate, like (laughs) or (sighs)."}
        ]
        for msg in chat_history:
            if msg['role'] in ['user', 'assistant']:
                messages.append(msg)
        
        terminators = [
            self.llm_pipeline.tokenizer.eos_token_id,
            self.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.llm_pipeline(
            messages, max_new_tokens=256, eos_token_id=terminators, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
        )
        assistant_response = outputs[0]["generated_text"][-1]['content']
        print(f"Agent: {assistant_response}")
        return assistant_response

    def convert_text_to_speech(self, text: str) -> str:
        """Converts text to speech using the Dia model."""
        print("Speaking with Dia...")
        
        formatted_text = f"[S1] {text} [S1]"
        
        inputs = self.tts_processor(
            text=[formatted_text], 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.tts_model.generate(
                **inputs, 
                max_new_tokens=4096,
                guidance_scale=3.0, 
                temperature=1.0, 
                top_p=0.90, 
                top_k=45
            )

        decoded_outputs = self.tts_processor.batch_decode(outputs)
        audio_array = decoded_outputs[0]["audio"][0]
        samplerate = decoded_outputs[0]["sampling_rate"]
        
        sf.write(OUTPUT_WAV_FILE, audio_array, samplerate)
        
        return OUTPUT_WAV_FILE

    def process_conversation_turn(self, audio_filepath: str, chat_history: list):
        """Processes a single conversational turn."""
        if audio_filepath is None: return chat_history, None
        user_text = self.transcribe_audio(audio_filepath)
        if not user_text.strip(): return chat_history, None
            
        chat_history.append({"role": "user", "content": user_text})
        llm_response = self.generate_response(chat_history)
        chat_history.append({"role": "assistant", "content": llm_response})
        
        agent_audio_path = self.convert_text_to_speech(llm_response)
        return chat_history, agent_audio_path

def build_ui(agent: RealTimeS2SAgent):
    """Builds the Gradio web interface."""
    with gr.Blocks(theme=gr.themes.Soft(), title="S2S Agent with Dia TTS") as demo:
        gr.Markdown("# Real-Time Speech-to-Speech AI Agent (Dia TTS)")
        gr.Markdown("Tap the microphone, speak, and the agent will respond with an expressive voice.")

        chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot", height=500, type="messages")
        
        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Tap to Talk")
            audio_output = gr.Audio(label="Agent Response", autoplay=True, visible=True)

        def handle_interaction(audio_filepath, history):
            history = history or []
            return agent.process_conversation_turn(audio_filepath, history)

        mic_input.stop_recording(
            fn=handle_interaction,
            inputs=[mic_input, chatbot],
            outputs=[chatbot, audio_output]
        )
        
        clear_button = gr.Button("Clear Conversation")
        clear_button.click(lambda: ([], None), None, [chatbot, audio_output])

    return demo

if __name__ == "__main__":
    # This block contains the definitive fix for Gradio in containerized environments.
    
    # 1. Set the GRADIO_SERVER_NAME environment variable.
    #    This tells Gradio's internal health check to look for the server at 127.0.0.1,
    #    which is reliable inside a container.
    os.environ['GRADIO_SERVER_NAME'] = '127.0.0.1'
    
    # 2. Instantiate the agent and build the UI.
    agent = RealTimeS2SAgent()
    ui = build_ui(agent)
    
    # 3. Launch the server.
    #    - server_name="0.0.0.0" makes the server accessible from outside the container (needed for Runpod).
    #    - We do NOT use share=True or any other undocumented flags.
    ui.launch(server_name="0.0.0.0", server_port=7860)
