import torch
import transformers
import faster_whisper
import gradio as gr
import numpy as np
import soundfile as sf
import os

# Configuration
STT_MODEL = "distil-large-v3"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TTS_MODEL = "nari-labs/Dia-1.6B-0626"
OUTPUT_WAV_FILE = "output.wav"

class RealTimeS2SAgent:
    def __init__(self):
        print("--- Initializing S2S Agent with Dia TTS ---")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required.")
        
        self.device = "cuda"
        print(f"Using device: {self.device.upper()}")

        print(f"Loading STT: {STT_MODEL}...")
        self.stt_model = faster_whisper.WhisperModel(STT_MODEL, device=self.device, compute_type="float16")
        print("STT loaded.")

        print(f"Loading LLM: {LLM_MODEL}...")
        self.llm_pipeline = transformers.pipeline("text-generation", model=LLM_MODEL, model_kwargs={"torch_dtype": torch.bfloat16}, device_map=self.device)
        print("LLM loaded.")

        print(f"Loading Dia TTS: {TTS_MODEL}...")
        self.tts_processor = transformers.AutoProcessor.from_pretrained(TTS_MODEL)
        self.tts_model = transformers.DiaForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=torch.bfloat16).to(self.device)
        print("Dia TTS loaded.")
        
        if os.path.exists(OUTPUT_WAV_FILE):
            os.remove(OUTPUT_WAV_FILE)
        
        print("\n--- Agent Ready ---")
        self.speaker_tag = "[S1]"  # For alternating speakers if needed

    def transcribe_audio(self, audio_filepath: str) -> str:
        if not audio_filepath: return ""
        print("Transcribing...")
        segments, _ = self.stt_model.transcribe(audio_filepath, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        print(f"User: {transcription}")
        return transcription

    def generate_response(self, chat_history: list) -> str:
        messages = [{"role": "system", "content": "You are Deva, a friendly AI. Be concise, expressive. Use (laughs) or (sighs) where fitting."}]
        messages.extend([msg for msg in chat_history if msg['role'] in ['user', 'assistant']])
        
        terminators = [self.llm_pipeline.tokenizer.eos_token_id, self.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = self.llm_pipeline(messages, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=self.llm_pipeline.tokenizer.eos_token_id)
        response = outputs[0]["generated_text"][-1]['content']
        print(f"Agent: {response}")
        return response

    def convert_text_to_speech(self, text: str) -> str:
        print("Speaking with Dia...")
        formatted_text = f"{self.speaker_tag} {text} {self.speaker_tag}"
        # Alternate tag for next call (simple multi-speaker)
        self.speaker_tag = "[S2]" if self.speaker_tag == "[S1]" else "[S1]"
        
        inputs = self.tts_processor(text=[formatted_text], padding=True, return_tensors="pt").to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.tts_model.generate(**inputs, max_new_tokens=4096, guidance_scale=3.0, temperature=1.8, top_p=0.90, top_k=45)
            decoded = self.tts_processor.batch_decode(outputs)
            audio_array = decoded[0]["audio"][0]
            samplerate = decoded[0]["sampling_rate"]
            sf.write(OUTPUT_WAV_FILE, audio_array, samplerate)
            return OUTPUT_WAV_FILE
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def process_conversation_turn(self, audio_filepath: str, chat_history: list):
        if audio_filepath is None: return chat_history, None
        user_text = self.transcribe_audio(audio_filepath)
        if not user_text.strip(): return chat_history, None
        
        chat_history.append({"role": "user", "content": user_text})
        llm_response = self.generate_response(chat_history)
        chat_history.append({"role": "assistant", "content": llm_response})
        
        agent_audio_path = self.convert_text_to_speech(llm_response)
        return chat_history, agent_audio_path

def build_ui(agent: RealTimeS2SAgent):
    with gr.Blocks(theme=gr.themes.Soft(), title="S2S Agent with Dia TTS") as demo:
        gr.Markdown("# Real-Time Speech-to-Speech AI Agent (Dia TTS)")
        gr.Markdown("Tap mic, speak; agent responds expressively.")

        chatbot = gr.Chatbot(label="Conversation", height=500, type="messages")
        
        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Tap to Talk")
            audio_output = gr.Audio(label="Agent Response", autoplay=True, visible=True)

        def handle_interaction(audio_filepath, history):
            history = history or []
            return agent.process_conversation_turn(audio_filepath, history)

        mic_input.stop_recording(fn=handle_interaction, inputs=[mic_input, chatbot], outputs=[chatbot, audio_output])
        
        clear_button = gr.Button("Clear Conversation")
        clear_button.click(lambda: ([], None), None, [chatbot, audio_output])

    return demo

if __name__ == "__main__":
    os.environ['GRADIO_SERVER_NAME'] = '127.0.0.1'
    agent = RealTimeS2SAgent()
    ui = build_ui(agent)
    ui.launch(server_name="0.0.0.0", server_port=7860, show_api=False)  # Disable API to avoid schema error
