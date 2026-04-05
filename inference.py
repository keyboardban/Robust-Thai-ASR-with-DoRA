import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# ==========================================
# 1. ตั้งค่า Model IDs
# ==========================================
BASE_MODEL_ID = "nectec/Pathumma-whisper-th-large-v3"
PEFT_MODEL_ID = "pmootr/pathumma-large-v3-dora-robust"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"📥 Loading Base Model on {DEVICE}...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, device_map=DEVICE)
    
    print("🧩 Loading and Merging DoRA Weights from Hugging Face...")
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
    
    # หลอมรวม Adapter เข้ากับ Base model เพื่อลดการใช้ Memory และเพิ่มความเร็ว
    model = model.merge_and_unload()
    print("✅ Model is ready!")
    return processor, model

def transcribe(audio_path, processor, model):
    print(f"🎙️ Transcribing: {audio_path}")
    # Load audio (บังคับ Sampling Rate ที่ 16000Hz สำหรับ Whisper)
    audio_array, sr = librosa.load(audio_path, sr=16000)
    
    inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="thai", task="transcribe")
    
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs.input_features.to(DEVICE, dtype=model.dtype),
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=255,
            num_beams=5,
            repetition_penalty=1.2
        )
        
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. โหลดฟังก์ชันทำความสะอาดข้อความ จากไฟล์ postprocess.py ของเรา
    from postprocess import clean_output 
    
    processor, model = load_model()
    
    # 2. ใส่ไฟล์เสียงที่ต้องการทดสอบ
    sample_audio = "test_audio.wav" 
    
    print("-" * 50)
    # 3. ถอดเสียงแบบดิบๆ (Raw)
    raw_text = transcribe(sample_audio, processor, model)
    print(f"📝 Raw Transcription   : {raw_text}")
    
    # 4. ฟอกข้อความให้สมบูรณ์ (Cleaned)
    final_text = clean_output(raw_text)
    print(f"✨ Cleaned Transcription: {final_text}")
    print("-" * 50)