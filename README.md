```markdown
# 🎙️ Thai ASR Optimization with DoRA & Pathumma-Whisper

This repository contains the inference code and post-processing pipeline for a highly robust Thai Automatic Speech Recognition (ASR) model. The project was developed to tackle extreme acoustic conditions, including heavy reverberation and poor-quality microphones.

## 🔗 Hugging Face Model Weights
The fine-tuned DoRA adapter weights are hosted on Hugging Face. We highly recommend downloading the weights directly from there:
👉 **[Hugging Face: pmootr/pathumma-large-v3-dora-robust](https://huggingface.co/pmootr/pathumma-large-v3-dora-robust)**

## 🏆 Project Highlights
* **Base Model:** `nectec/Pathumma-whisper-th-large-v3`
* **Dataset:** Evaluated and trained on a curated "Golden Dataset" derived from the **LOTUSDIS dataset (NECTEC)**. Heavily corrupted audio was filtered out using WER evaluation.
* **Architecture:** Utilized **DoRA (Weight-Decomposed Low-Rank Adaptation)** targeting `all_linear` layers for optimal performance without full fine-tuning.
* **Data Strategy:** Applied Stratified Sampling to balance 6 different microphone profiles.
* **Post-Processing:** Integrated `PyThaiNLP` to normalize Thai text (fixing floating vowels) and convert Arabic numerals to Thai words.

## ⚙️ Repository Structure
* `inference.py`: Script to load the base model, attach the DoRA weights from Hugging Face, and transcribe audio.
* `postprocess.py`: Script containing the Rule-based and PyThaiNLP text normalization pipeline.
* `requirements.txt`: Python dependencies.

## 🚀 Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
2. Run the inference script
   python inference.py