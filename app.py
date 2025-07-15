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

        print(f"Loading STT model: {STT_MODEL}...")
        self.stt_model = faster_whisper.WhisperModel(
            STT_MODEL, 
            device=self.device, 
            compute_type="float16"
        )
        print("STT model loaded.")

        print(f"Loading LLM: {LLM_MODEL}...")
        self.llm_pipeline = transformers.pipeline(
            "text-generation",
            model=LLM_MODEL,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self.device,
        )
        print("LLM loaded.")

        print(f"Loading Dia TTS model: {TTS_MODEL}...")
        self.tts_processor = transformers.AutoProcessor.from_pretrained(TTS_MODEL)
        self.tts_model = transformers.DiaForConditionalGeneration.from_pretrained(
            TTS_MODEL, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        print("Dia TTS model loaded.")
        
        # Speaker tag state for alternating dialogue
        self.speaker_tag = "[S1]"
        
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
        """Generates a response from the LLM with a persona-driven system prompt."""
        
        # This new system prompt is the key to fixing the AI's behavior.
        # It explicitly tells the LLM it has a voice and how to behave.
        system_prompt = (
            "You are Deva, a friendly and expressive AI assistant. Your responses will be converted into "
            "audible speech, so your primary goal is to generate text that sounds natural and engaging when spoken. "
            "Use conversational language and non-verbal cues like (laughs), (sighs), or (ahem) to add personality. "
            "Crucially, always act as though you are speaking. Never describe yourself as a text-based model or "
            "mention that you cannot speak. You have a voice."
        )

        messages = [{"role": "system", "content": system_prompt}]
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
        """Converts text to speech using the Dia model and its processor's save function."""
        print("Speaking with Dia...")
        
        # Dia works best with alternating speaker tags. We'll manage this state.
        formatted_text = f"{self.speaker_tag} {text} {self.speaker_tag}"
        # Alternate the speaker tag for the next turn to make dialogue more natural
        self.speaker_tag = "[S2]" if self.speaker_tag == "[S1]" else "[S1]"
        
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

        # Use the processor's built-in save function for robust audio saving
        decoded_outputs = self.tts_processor.batch_decode(outputs)
        self.tts_processor.save_audio(decoded_outputs, OUTPUT_WAV_FILE)
        
        return OUTPUT_WAV_FILE

    def process_conversation_turn(self, audio_filepath: str, chat_history: list):
        """Processes a single conversational turn from audio input to audio output."""
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
            """Simplified callback function."""
            history = history or []
            return agent.process_conversation_turn(audio_filepath, history)

        mic_input.stop_recording(
            fn=handle_interaction,
            inputs=[mic_input, chatbot],
            outputs=[chatbot, audio_output]
        )
        
        def clear_chat():
            # Reset the speaker tag when clearing the chat
            agent.speaker_tag = "[S1]"
            return [], None

        clear_button = gr.Button("Clear Conversation")
        clear_button.click(fn=clear_chat, outputs=[chatbot, audio_output], queue=False)

    return demo

if __name__ == "__main__":
    # This block contains the definitive fix for Gradio in containerized environments.
    
    # 1. Set the GRADIO_SERVER_NAME environment variable.
    os.environ['GRADIO_SERVER_NAME'] = '127.0.0.1'
    
    # 2. Instantiate the agent and build the UI.
    agent = RealTimeS2SAgent()
    ui = build_ui(agent)
    
    # 3. Launch the server.
    ui.launch(server_name="0.0.0.0", server_port=7860)
