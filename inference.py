import torch, torchaudio, os, argparse
import numpy as np
from ecapa_model import ECAPA_TDNN # Model dosyasını çağır

def load_model(model_path, C=256):
    # Modeli başlat (Backbone sadece, Classifier'a gerek yok)
    model = ECAPA_TDNN(C=C).cuda()
    
    # Checkpoint'i yükle
    # Not: Kaydettiğimiz dosya tüm durumu (optimizer vs) içeriyor olabilir
    # bu yüzden sadece state_dict içindeki "speaker_encoder" kısmını alacağız.
    print(f"Model yükleniyor: {model_path}")
    loaded_state = torch.load(model_path)
    
    # Parametrelerin isimlerini eşleştir (prefix temizliği)
    self_state = model.state_dict()
    for name, param in loaded_state.items():
        # "speaker_encoder." prefixini kaldır
        name = name.replace("speaker_encoder.", "")
        if name in self_state:
            self_state[name].copy_(param)
            
    model.eval()
    return model

def compute_embedding(model, audio_path, num_frames=200):
    # Sesi yükle
    audio, sr = torchaudio.load(audio_path)
    
    # Resample (Eğer 24k değilse 24k yap)
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
        audio = resampler(audio)
        
    # Test sırasında sesi rastgele kesmek yerine,
    # belirli bir uzunluğu veya tamamını alabiliriz.
    # Kodu basit tutmak için train'deki mantığın aynısını (loop/pad) uygulayalım.
    
    # Sabit uzunluk hesabı (24k için)
    # length = num_frames * 240 + 360
    # Ancak testte tüm sesi vermek daha mantıklıdır. 
    # ECAPA değişken uzunluk kabul edebilir.
    
    # Sadece batch boyutu ekle (1, Samples)
    audio = audio.cuda()
    
    with torch.no_grad():
        embedding = model(audio, aug=False) # Augmentasyon kapalı
        
    return embedding

def compare_speakers(model, file1, file2):
    emb1 = compute_embedding(model, file1)
    emb2 = compute_embedding(model, file2)
    
    # Cosine Similarity
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    score = similarity(emb1, emb2)
    
    return score.item()

if __name__ == "__main__":
    # AYARLAR
    MODEL_PATH = "exps/mini_ecapa_tts/best_model.model" # Best model yolu
    AUDIO_1 = "deneme_sesleri/benim_sesim_1.wav"
    AUDIO_2 = "deneme_sesleri/benim_sesim_2.wav"
    AUDIO_3 = "deneme_sesleri/baska_bir_adam.wav"
    
    # Model Yükle
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, C=256)
        
        print("-" * 30)
        
        # Test 1: Aynı Kişi (Sen vs Sen)
        if os.path.exists(AUDIO_1) and os.path.exists(AUDIO_2):
            score = compare_speakers(model, AUDIO_1, AUDIO_2)
            print(f"Benim Sesim 1 vs Benim Sesim 2\nBenzerlik Skoru: {score:.4f}")
            print("Yorum: " + ("AYNI KİŞİ (Başarılı)" if score > 0.4 else "FARKLI KİŞİ (Başarısız)"))
        
        print("-" * 30)
        
        # Test 2: Farklı Kişi (Sen vs Başkası)
        if os.path.exists(AUDIO_1) and os.path.exists(AUDIO_3):
            score = compare_speakers(model, AUDIO_1, AUDIO_3)
            print(f"Benim Sesim 1 vs Başkası\nBenzerlik Skoru: {score:.4f}")
            print("Yorum: " + ("FARKLI KİŞİ (Başarılı)" if score < 0.4 else "AYNI KİŞİ (Hatalı)"))
            
    else:
        print("Model dosyası bulunamadı, önce eğitimi tamamla.")