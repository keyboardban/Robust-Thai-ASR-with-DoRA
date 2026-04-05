import re
from pythainlp.util import normalize, num_to_thaiword

def clean_output(text):
    """
    ฟังก์ชันทำความสะอาดและจัดรูปแบบข้อความภาษาไทย 
    ผ่านการทดสอบในการแข่งขัน Kaggle ASR
    """
    if not text or str(text).strip() == "":
        return " "
        
    text = str(text)
    
    # 1. Rule-based: ลบคำขยะ (Filler words) ที่ Whisper ชอบแถมมา
    filler_words = ["อ่า ", "เอ่อ ", "แบบว่า ", "คือว่า "]
    for word in filler_words:
        text = text.replace(word, "")
        
    # 2. Rule-based: แก้คำทับศัพท์ที่มักสะกดผิด
    replace_dict = {
        "OK": "โอเค",
        "AI": "เอไอ",
        "เปอร์เซ็น": "เปอร์เซ็นต์",
        "แอพ": "แอป"
    }
    for wrong, right in replace_dict.items():
        text = text.replace(wrong, right)
        
    # 3. PyThaiNLP: แก้ปัญหาสระลอย วรรณยุกต์ซ้อน (สำคัญมากสำหรับ WER)
    text = normalize(text)
    
    # 4. PyThaiNLP: แปลงเลขอารบิกเป็นคำอ่านภาษาไทย (เช่น 150 -> ร้อยห้าสิบ)
    def replace_number(match):
        return num_to_thaiword(int(match.group()))
    text = re.sub(r'\d+', replace_number, text)
    
    # 5. จัดการช่องว่างที่ซ้ำซ้อนให้เหลือเคาะเดียว
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text != "" else " "

# ==========================================
# Example Test
# ==========================================
if __name__ == "__main__":
    test_sentences = [
        "สวัสดีครับ อ่า วันนี้ มีผู้ร่วมงาน 150 คนเปอร์เซ็น",
        "เอ่อ OK ระบบ AI ทำงานปกติ",
        "นี่คือ ปัญหา สระลอย น้ำาา"
    ]
    
    print("--- 🧹 Text Post-Processing Test ---")
    for sentence in test_sentences:
        cleaned = clean_output(sentence)
        print(f"Original : {sentence}")
        print(f"Cleaned  : {cleaned}\n")