# 🎮 SonicGen: Audio-Driven Player DNA & Soundtrack AI
**GamerZone- | Foundation Models for Speech, Music, and Sound AI**

## 📖 Overview
Modern video games typically rely on static soundtracks and analyze player behavior through raw telemetry data (e.g., K/D ratios, movement logs). **SonicGen** bridges the gap between player emotion and game environments by introducing an audio-driven multimodal AI pipeline. 

This project captures simulated player voice communications, extracts an "8-Axis Gamer DNA" profile to understand behavior and intent, and dynamically generates custom background music tailored to their exact playstyle.

## 🗂️ Repository Contents
* **`Week_14_Hands_On_Challenge.ipynb`**: The core Python notebook executing the multimodal AI workflow. 
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
