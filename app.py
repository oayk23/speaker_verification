import gradio as gr
import torch
import torchaudio
import os
import numpy as np
from ecapa_model import ECAPA_TDNN 

# --- AYARLAR ---
MODEL_PATH = "artifacts\model_0057.model" # Model yolunu kontrol et!
THRESHOLD = 0.30  # Senin testlerinde bulduğun optimal eşik (0.2993)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODELİ YÜKLE ---
def load_model():
    print("Model yükleniyor...")
    model = ECAPA_TDNN(C=256).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        new_state = {}
        for k, v in checkpoint.items():
            if k.startswith("speaker_encoder."):
                new_state[k.replace("speaker_encoder.", "")] = v
        model.load_state_dict(new_state)
        model.eval()
        print("✅ Model hazır!")
        return model
    else:
        print(f"❌ HATA: Model dosyası bulunamadı: {MODEL_PATH}")
        return None

model = load_model()

# --- GELİŞMİŞ SES İŞLEME FONKSİYONU ---
def preprocess_audio(wav_path):
    """
    Sesi yükler, resample yapar, mono'ya çevirir, sessizliği atar ve normalize eder.
    """
    try:
        # 1. Yükle
        wav, sr = torchaudio.load(wav_path)
        
        # 2. Resample (24kHz)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            wav = resampler(wav)
            
        # 3. Mono Yap
        if wav.shape[0] > 1: 
            wav = torch.mean(wav, dim=0, keepdim=True)

        # 4. Basit Sessizlik Temizleme (Trim Silence)
        # Enerjisi çok düşük olan baştaki ve sondaki kısımları at
        # (Basit bir yöntem: Mutlak değer ortalamasının %10'u altını sessizlik say)
        threshold = wav.abs().mean() * 0.1
        mask = wav.abs() > threshold
        # Mask'in True olduğu ilk ve son indexi bul
        indices = torch.nonzero(mask)
        
        if indices.numel() > 0:
            start = indices.min()
            end = indices.max()
            wav = wav[:, start:end+1]
        
        # Eğer çok kısa kaldıysa (örn: sadece gürültü varsa) orijinali kullan
        if wav.shape[1] < 2400: # 0.1 saniyeden kısaysa
            wav, _ = torchaudio.load(wav_path) # Başa dön
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            if sr != 24000: wav = torchaudio.transforms.Resample(sr, 24000)(wav)

        # 5. Peak Normalization (En önemlisi bu!)
        # Sesi -1 ile +1 arasına yayar
        max_val = torch.abs(wav).max()
        if max_val > 0:
            wav = wav / max_val
            
        return wav.to(DEVICE)
        
    except Exception as e:
        print(f"Preprocess Hatası: {e}")
        return None

def get_embedding(wav_tensor):
    with torch.no_grad():
        # Batch boyutu ekle [1, samples]
        if wav_tensor.dim() == 2:
            wav_tensor = wav_tensor
        else:
            wav_tensor = wav_tensor.unsqueeze(0)
            
        emb = model(wav_tensor, aug=False)
    return emb

def compare_speakers(audio1, audio2):
    if model is None:
        return 0, "Model Yüklenemedi!", "error"
    
    if not audio1 or not audio2:
        return 0, "Lütfen iki ses dosyasını da yükleyin/kaydedin."
    
    # İşle
    wav1 = preprocess_audio(audio1)
    wav2 = preprocess_audio(audio2)
    
    if wav1 is None or wav2 is None:
        return 0, "Ses işlenirken hata oluştu."

    # Embedding Al
    emb1 = get_embedding(wav1)
    emb2 = get_embedding(wav2)
    
    # Skorla
    score = torch.nn.CosineSimilarity(dim=-1)(emb1, emb2).item()
    
    # Sonuç Metni
    score_display = float(f"{score:.4f}")
    
    if score > THRESHOLD:
        result_text = f"✅ AYNI KİŞİ (Eşleşti)"
        # Güven seviyesi ekleyelim
        confidence = min((score - THRESHOLD) / (1 - THRESHOLD) * 100 + 50, 99)
        result_text += f"\nGüven: %{confidence:.1f}"
    elif score > (THRESHOLD - 0.10):
        result_text = f"🤔 BELİRSİZ (Gri Alan)"
    else:
        result_text = f"❌ FARKLI KİŞİ"
        
    return score_display, result_text

# --- ARAYÜZ ---
css = """
#result_box {
    font-size: 22px; 
    font-weight: bold; 
    text-align: center;
    padding: 20px;
}
"""

with gr.Blocks(title="Mini-ECAPA Demo", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(
        """
        # 🎤 Mini-ECAPA: Türkçe Ses Doğrulama
        **Model:** Custom Mini-ECAPA (16MB) | **Eğitim:** 500 Saat Türkçe Veri | **Performans:** %0.56 EER
        """
    )
    
    with gr.Row():
        with gr.Column():
            audio_input1 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Ses 1 (Referans)")
        with gr.Column():
            audio_input2 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Ses 2 (Test)")
            
    submit_btn = gr.Button("🔍 Karşılaştır", variant="primary", size="lg")
    
    with gr.Row():
        score_output = gr.Number(label="Benzerlik Skoru", precision=4)
        text_output = gr.Textbox(label="Sonuç", elem_id="result_box")
    
    submit_btn.click(
        fn=compare_speakers,
        inputs=[audio_input1, audio_input2],
        outputs=[score_output, text_output]
    )
    
    gr.Markdown("--- \n *Geliştirilmiş Preprocessing (Normalization + Trim) Aktif*")

if __name__ == "__main__":
    demo.launch(share=True)