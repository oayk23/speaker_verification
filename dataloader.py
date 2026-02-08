'''
DataLoader for training - 24kHz Fixed
'''

import os, random, soundfile, torch, numpy, csv
import pandas as pd

class train_loader(torch.utils.data.Dataset):
    def __init__(self, dataset_csv_path, dataset_root_path, num_frames, speaker_dict, sample_rate=24000, **kwargs):
        self.num_frames = num_frames
        self.speaker_dict = speaker_dict
        self.sample_rate = sample_rate
        
        # 24kHz vs 16kHz ayarları
        if self.sample_rate == 24000:
            self.hop_length = 240
            self.context_offset = 360 
        elif self.sample_rate == 16000:
            self.hop_length = 160
            self.context_offset = 240
        else:
            self.hop_length = int(self.sample_rate * 0.010)
            win_length = int(self.sample_rate * 0.025)
            self.context_offset = win_length - self.hop_length

        self.data_list  = []
        self.data_label = []
        
        # --- DÜZELTME: Pandas ile okuma ---
        print(f"Veri yükleniyor: {dataset_csv_path}")
        try:
            df = pd.read_csv(dataset_csv_path,sep="|")
            # Sütun isimlerindeki boşlukları temizle
            df.columns = [c.strip() for c in df.columns]
            
            # Sütun isimlerini belirle (Otomatik bulmaya çalış)
            if 'speaker' in df.columns:
                spk_col = 'speaker'
            else:
                spk_col = df.columns[2] # İsim yoksa 3. sütun

            if 'audio_path' in df.columns:
                aud_col = 'audio_path'
            else:
                aud_col = df.columns[0] # İsim yoksa 1. sütun

            # Verileri listeye ekle
            loaded_count = 0
            missing_speakers = 0
            
            # DataFrame'i hızlıca dön
            for audio_filename, speaker_name in zip(df[aud_col], df[spk_col]):
                # Speaker global sözlükte var mı?
                if speaker_name in self.speaker_dict:
                    full_path = os.path.join(dataset_root_path, str(speaker_name), str(audio_filename))
                    label = self.speaker_dict[speaker_name]
                    
                    self.data_label.append(label)
                    self.data_list.append(full_path)
                    loaded_count += 1
                else:
                    missing_speakers += 1

            print(f" -> {dataset_csv_path}: {loaded_count} dosya başarıyla eklendi.")
            if missing_speakers > 0:
                print(f" -> UYARI: {missing_speakers} satır, konuşmacı adı sözlükte bulunamadığı için atlandı.")
            
            if loaded_count == 0:
                print(" -> HATA: Hiçbir dosya yüklenemedi! CSV içeriğini, speaker sütununu veya 'root_path'i kontrol et.")
                print(f"    Örnek Speaker (CSV): {df[spk_col].iloc[0] if len(df)>0 else 'Yok'}")
                print(f"    Sözlükteki Örnek: {list(self.speaker_dict.keys())[0] if len(self.speaker_dict)>0 else 'Boş'}")

        except Exception as e:
            print(f"CSV Okuma Hatası: {e}")
            raise e
    def __getitem__(self, index):
        try:
            audio, sr = soundfile.read(self.data_list[index])
            # Ses 24k değilse bile okur ama ideal olan verinin 24k olmasıdır.
        except Exception as e:
            print(f"Hata dosya: {self.data_list[index]}")
            # Hata durumunda rastgele başka bir dosya seçip onu döndürebiliriz
            # Şimdilik hata verip dursun.
            raise e
        
        # --- DÜZELTİLMİŞ UZUNLUK HESABI ---
        length = self.num_frames * self.hop_length + self.context_offset
        
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        
        audio = numpy.stack([audio], axis=0)
        
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)