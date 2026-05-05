# 🎮 SonicGen: Audio-Driven Player DNA & Soundtrack AI
**GamerZone- | Foundation Models for Speech, Music, and Sound AI**

## 📖 Overview
Modern video games typically rely on static soundtracks and analyze player behavior through raw telemetry data (e.g., K/D ratios, movement logs). **SonicGen** bridges the gap between player emotion and game environments by introducing an audio-driven multimodal AI pipeline. 

This project captures simulated player voice communications, extracts an "8-Axis Gamer DNA" profile to understand behavior and intent, and dynamically generates custom background music tailored to their exact playstyle.

## 🗂️ Repository Contents
* **`Week_14_Hands_On_Challenge.ipynb/week_14_hands_on_challenge.py`**: The core Python notebook executing the multimodal AI workflow. 
* **`SonicGen_Presentation.pdf`**: A comprehensive slide deck detailing the problem, pipeline architecture, prompt engineering strategies, and evaluation metrics.
* **`Output/`**: Directory containing the generated audio files and sample datasets demonstrating the pipeline's end-to-end functionality.

## 🏗️ Pipeline Architecture
The system employs a multimodal chaining strategy, built entirely in Python, utilizing state-of-the-art foundation models:
1. **Phase 1: Speech-to-Text (`openai/whisper-small`)**
   - Ingests player voice chat audio and generates a highly accurate transcription, successfully filtering out standard background noise.
2. **Phase 2: DNA Extraction Logic**
   - Analyzes the transcribed text to classify the player's current behavior into an "8-Axis Gamer DNA" trait (e.g., Aggressive/Rush vs. Stealth/Tactical).
3. **Phase 3: Text-to-Music (`facebook/musicgen-small`)**
   - Takes the dynamically generated prompt based on the DNA profile (e.g., *"Aggressive cyberpunk battle music, 140bpm"*) and synthesizes a custom soundtrack matching the player's real-time emotional state.

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone [https://github.com/PrathyushaRagavAdari/GamerZone-.git](https://github.com/PrathyushaRagavAdari/GamerZone-.git)
   ```
   
   or

1. Open the included `Week_14_Hands_On_Challenge.ipynb` in Google Colab.
2. Enable a T4 GPU (Runtime > Change runtime type).
3. Run all cells. The script will download the models, process simulated audio, and output a `custom_player_soundtrack.wav` file.

## Demo Video 
https://umsystem.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=8111bcfa-f28d-4a86-9a13-b43a00481e6e

#
# 🎮 SonicGen v2.0: Multimodal Player DNA Studio
**Foundation Models for Speech, Music, and Sound AI**

## 🚀 Overview (Final Enhancements)
SonicGen v2.0 is a massive upgrade to the original audio-driven pipeline. It is now a fully interactive Web App that ingests player voice chat, extracts behavioral DNA, filters for toxicity, and outputs **both** dynamically generated music AND an AI companion voice response.

### 🌟 New Features & Enhancements
* **Interactive UI:** Fully functional web application built with Gradio.
* **Advanced Pipeline:** Upgraded from (Speech $\rightarrow$ Text $\rightarrow$ Music) to (Speech $\rightarrow$ Text $\rightarrow$ Toxicity Logic $\rightarrow$ Music + TTS Voice Response).
* **Real-time Evaluation:** On-the-fly calculation of Word Error Rate (WER) and Pipeline Latency.
* **Responsible AI:** Integrated toxicity filtering to prevent the AI from rewarding abusive gaming chat with aggressive soundtracks.
* **Optimized Engineering:** Implemented FP16 precision on GPU to drastically reduce Whisper inference time.

## 🏗️ Architecture Stack
* **Speech-to-Text:** `openai/whisper-small`
* **Text-to-Music:** `facebook/musicgen-small`
* **Text-to-Speech:** `gTTS` (Google Text-to-Speech)
* **Frontend/UI:** `Gradio`
* **Evaluation:** `jiwer` (Word Error Rate)

## 🗂️ Repository Contents
* **`main.py`**: The enhanced core Python notebook executing the multimodal AI workflow. 
* **`SonicGen_v2.pdf`**: An enhanced comprehensive slide deck detailing the problem, pipeline architecture, prompt engineering strategies, and evaluation metrics.
* **`Output/`**: Directory containing updated files demonstrating the enhanced pipeline's end-to-end functionality.

## ⚙️ How to Run the App
1. Open the included `.ipynb` notebook in Google Colab.
2. Navigate to `Runtime > Change runtime type` and select **T4 GPU**.
3. Run the single code cell.
4. Click the public **Gradio Live URL** generated at the bottom of the cell to open the Web App in your browser.
5. Use your microphone to record a gaming command (e.g., "Push the base now!") and watch the pipeline work.

## 📈 Business Value
SonicGen provides a blueprint for SaaS integration into platforms like Discord or OBS, offering streamers copyright-free, emotion-driven background music, and offering indie game developers a way to increase player retention through deeper audio immersion.

## Demo Video 
