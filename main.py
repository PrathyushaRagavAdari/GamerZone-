# -*- coding: utf-8 -*-
"""main

Original file is located at
    https://colab.research.google.com/drive/1o0W8OuM3z5aEuFcHWQzg9Xr306wLspg4
"""

# 1. Install Required Libraries
!pip install transformers accelerate scipy datasets librosa gradio jiwer gTTS > /dev/null

import torch
import time
import gradio as gr
import scipy.io.wavfile
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
from jiwer import wer
from gtts import gTTS
import os

print("Initializing Advanced SonicGen Pipeline...")

# 2. Hardware Optimization (Requirement 7: GPU Optimization)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 3. Load Models (Requirement 1: Faster Inference via fp16)
print("Loading Whisper...")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if device == "cuda" else -1, torch_dtype=dtype)

print("Loading MusicGen...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

# 4. Responsible AI & Logic (Requirement 6)
TOXIC_WORDS = ["hate", "kill yourself", "slur"] # Simplified for demo

def analyze_dna_and_ethics(transcription):
    text = transcription.lower()

    # Ethics Check
    if any(word in text for word in TOXIC_WORDS):
        return "TOXICITY DETECTED", "Calm ambient music", "I cannot process toxic language."

    # DNA Extraction
    if any(word in text for word in ["push", "attack", "fast", "rush"]):
        dna = "Aggressive/Rush"
        music_prompt = "Heavy bass, fast tempo 140 bpm, aggressive cyberpunk synthesizer"
        agent_response = "Aggressive push detected. Generating high-tempo battle music."
    elif any(word in text for word in ["hide", "wait", "quiet", "stealth"]):
        dna = "Stealth/Tactical"
        music_prompt = "Dark ambient, tense strings, slow tempo, quiet atmospheric thriller"
        agent_response = "Stealth mode engaged. Lowering audio profile."
    else:
        dna = "Casual/Exploration"
        music_prompt = "Lo-fi chill gaming beats, relaxing, peaceful synthesizer"
        agent_response = "Exploration phase. Playing relaxing background tracks."

    return dna, music_prompt, agent_response

# 5. Core Pipeline Function
def process_player_audio(audio_filepath, true_transcript=""):
    start_time = time.time()

    # Step A: Speech to Text
    transcription = transcriber(audio_filepath)["text"]

    # Step B: Evaluation (Requirement 4: WER & Latency)
    error_rate = wer(true_transcript.lower(), transcription.lower()) if true_transcript else 0.0

    # Step C: DNA & Ethics Logic
    dna, music_prompt, agent_text = analyze_dna_and_ethics(transcription)

    if dna == "TOXICITY DETECTED":
        return transcription, dna, agent_text, None, None, f"Latency: {round(time.time()-start_time, 2)}s", "N/A"

    # Step D: Music Generation (Hyperparameter tuned for faster, 5-second bursts)
    inputs = processor(text=[music_prompt], padding=True, return_tensors="pt").to(device)
    audio_values = music_model.generate(**inputs, max_new_tokens=256) # Approx 5 seconds

    sampling_rate = music_model.config.audio_encoder.sampling_rate
    music_data = audio_values[0, 0].cpu().numpy()
    music_output_path = "generated_music.wav"
    scipy.io.wavfile.write(music_output_path, rate=sampling_rate, data=music_data)

    # Step E: Advanced Pipeline (Text to Speech for Agent Voice)
    tts = gTTS(text=agent_text, lang='en', slow=False)
    voice_output_path = "agent_voice.mp3"
    tts.save(voice_output_path)

    end_time = time.time()
    latency = round(end_time - start_time, 2)

    return transcription, dna, music_prompt, music_output_path, voice_output_path, f"{latency} seconds", f"{round(error_rate * 100, 2)}%"

# 6. Gradio UI (Requirement 3: UI / Productization)
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🎮 SonicGen v2.0: Multimodal Player DNA Studio")
    gr.Markdown("Upload or record gaming voice chat to generate a personalized behavioral profile, custom soundtrack, and AI agent voice response.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Player Voice Input")
            true_text = gr.Textbox(label="True Transcript (Optional, for WER Evaluation)", placeholder="Type exactly what you said to calculate error rate...")
            submit_btn = gr.Button("Analyze & Generate", variant="primary")

        with gr.Column():
            out_transcription = gr.Textbox(label="Whisper Transcription")
            out_dna = gr.Textbox(label="Extracted Gamer DNA")
            out_prompt = gr.Textbox(label="MusicGen Prompt")

    with gr.Row():
        out_voice = gr.Audio(label="AI Companion Response (Speech output)")
        out_music = gr.Audio(label="Generated Custom Soundtrack")

    with gr.Row():
        out_latency = gr.Textbox(label="Pipeline Latency")
        out_wer = gr.Textbox(label="Word Error Rate (WER)")

    submit_btn.click(
        fn=process_player_audio,
        inputs=[audio_input, true_text],
        outputs=[out_transcription, out_dna, out_prompt, out_music, out_voice, out_latency, out_wer]
    )

print("Launching UI...")
demo.launch(share=True) # Generates a public web link
