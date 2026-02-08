'''
Main code for ECAPA-TDNN training with Auto-Split Stratified Logic
'''

import argparse, glob, os, torch, warnings, time, csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import train_loader
from ecapa_model import ECAPAModel

parser = argparse.ArgumentParser(description = "ECAPA_trainer")

## 1. Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Input length (200 frames)')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=128,      help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=0,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay')

## 2. Paths (TEK CSV YOLU)
parser.add_argument('--main_csv',   type=str,   default="metadata.csv",         help='Path to your single main CSV file')
parser.add_argument('--root_path',  type=str,   default="dataset",              help='Root directory of audio files')
parser.add_argument('--save_path',  type=str,   default="exps/mini_ecapa_tts",  help='Path to save models')
parser.add_argument('--initial_model', type=str, default="",                    help='Path of the initial_model to resume')

## 3. Model Parameters (Mini Voice Cloning Config)
parser.add_argument('--C',          type=int,   default=256,    help='Channel size (256 for mini model)')
parser.add_argument('--m',          type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',          type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class',    type=int,   default=0,      help='Will be calculated automatically')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# --------------------------------------------------------------------------
# ADIM 1: Veriyi Okuma ve Akıllı Bölme (Stratified Split)
# --------------------------------------------------------------------------

def prepare_data_split(csv_path, save_dir):
    print(f"CSV okunuyor: {csv_path}")
    df = pd.read_csv(csv_path,sep="|")
    
    # CSV sütun isimlerini kontrol et (audio_path, text, speaker, duration) bekliyoruz
    # Garanti olsun diye sütun isimlerini temizleyelim
    df.columns = [c.strip() for c in df.columns]
    
    # Speaker kolonunun adı 'speaker' olmalı. Değilse 3. kolon varsayalım.
    if 'speaker' not in df.columns:
        # Eğer header yoksa veya farklıysa manuel index kullanırız ama senin formatında header var.
        # Yine de güvenli olsun diye:
        speaker_col = df.columns[2] 
    else:
        speaker_col = 'speaker'

    print(f"Toplam Veri: {len(df)} satır")
    print(f"Toplam Konuşmacı: {df[speaker_col].nunique()}")

    # Az örneği olan (1 tane) konuşmacıları bul
    speaker_counts = df[speaker_col].value_counts()
    single_sample_speakers = speaker_counts[speaker_counts < 2].index.tolist()
    
    # Sadece 1 örneği olanları direkt TRAIN'e ayır (Validation'a giderse hata verir)
    train_force = df[df[speaker_col].isin(single_sample_speakers)]
    remaining_df = df[~df[speaker_col].isin(single_sample_speakers)]
    
    print(f"Tek örneği olan {len(train_force)} satır doğrudan Train setine eklendi.")

    # Geri kalanları %90 Train, %10 Val olarak böl (Stratified)
    # Stratify: Her konuşmacıdan orantılı olarak val setine koyar.
    if len(remaining_df) > 0:
        X_train, X_val = train_test_split(
            remaining_df, 
            test_size=0.10, 
            random_state=42, 
            stratify=remaining_df[speaker_col]
        )
        # Zorunlu train verilerini ekle
        final_train = pd.concat([X_train, train_force])
        final_val = X_val
    else:
        # Eğer herkesin sadece 1 örneği varsa (imkansız ama) hepsi train olur
        final_train = train_force
        final_val = pd.DataFrame(columns=df.columns)
        print("UYARI: Validation seti oluşturulamadı, tüm veri Train yapıldı.")

    # Bölünmüş dosyaları kaydet
    train_csv_path = os.path.join(save_dir, "train_split.csv")
    val_csv_path = os.path.join(save_dir, "val_split.csv")
    
    final_train.to_csv(train_csv_path, index=False,sep="|")
    final_val.to_csv(val_csv_path, index=False,sep="|")
    
    print(f"Train Seti: {len(final_train)} örnek -> {train_csv_path}")
    print(f"Val Seti:   {len(final_val)} örnek -> {val_csv_path}")
    
    return train_csv_path, val_csv_path, df[speaker_col].nunique()

# CSV Ayırma İşlemini Yap
if not args.eval: # Sadece eğitim modunda split yapalım
    args.train_csv, args.val_csv, total_speakers = prepare_data_split(args.main_csv, args.save_path)
    args.n_class = total_speakers
    
    # Global Speaker Dictionary oluştur (ID'lerin sabit kalması için)
    # Pandas ile zaten okuduk, unique speakerları alıp sıralayalım
    full_df = pd.read_csv(args.main_csv,sep="|")
    # Sütun adı düzeltme (yukarıdaki gibi)
    spk_col = 'speaker' if 'speaker' in full_df.columns else full_df.columns[2]
    unique_speakers = sorted(full_df[spk_col].unique().tolist())
    global_speaker_dict = {spk: i for i, spk in enumerate(unique_speakers)}
    
else:
    # Eval modundaysa var olan splitleri veya ana dosyayı kullanabilirsin
    # Şimdilik eval için manuel path gerekebilir, burayı basit tutuyorum.
    pass

# --------------------------------------------------------------------------
# ADIM 2: DataLoader
# --------------------------------------------------------------------------

# Train Loader
train_dataset = train_loader(
    dataset_csv_path=args.train_csv,
    dataset_root_path=args.root_path,
    num_frames=args.num_frames,
    speaker_dict=global_speaker_dict,
    sample_rate=24000
)

trainLoader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=args.n_cpu, 
    drop_last=True
)

# Validation Loader
val_dataset = train_loader(
    dataset_csv_path=args.val_csv,
    dataset_root_path=args.root_path,
    num_frames=args.num_frames,
    speaker_dict=global_speaker_dict,
    sample_rate=24000
)

valLoader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=args.n_cpu, 
    drop_last=False
)

# --------------------------------------------------------------------------
# ADIM 3: Model & Eğitim
# --------------------------------------------------------------------------

## Model Arama
modelfiles = glob.glob('%s/model_0*.model'%args.save_path)
modelfiles.sort()

if args.initial_model != "":
    print("Model %s loaded from previous state!"%args.initial_model)
    s = ECAPAModel(**vars(args))
    s.load_parameters(args.initial_model)
    epoch = 1
elif len(modelfiles) >= 1:
    print("Model %s loaded from previous state!"%modelfiles[-1])
    epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0].split('_')[-1]) + 1
    s = ECAPAModel(**vars(args))
    s.load_parameters(modelfiles[-1])
else:
    epoch = 1
    s = ECAPAModel(**vars(args)).to("cuda")

score_file = open(os.path.join(args.save_path, "score.txt"), "a+")

print("Eğitim Başlıyor...")

best_val_loss = float('inf') # Sonsuz ile başlat
patience = 5                 # 5 epoch boyunca iyileşme olmazsa dur
patience_counter = 0

print("Eğitim Başlıyor...")

while(1):
    # 1. Eğitim
    loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)

    # 2. Validasyon ve Kayıt (Test Step)
    if epoch % args.test_step == 0:
        
        # Mevcut epoch modelini kaydet (her ihtimale karşı)
        s.save_parameters(args.save_path + "/model_%04d.model"%epoch)
        
        # Validasyon skorlarını al
        val_loss, val_acc = s.eval_network(loader=valLoader)
        
        print(time.strftime("%Y-%m-%d %H:%M:%S"), 
              "Epoch %d, Train ACC %2.2f%%, Val ACC %2.2f%%, Val Loss %.4f"
              %(epoch, acc, val_acc, val_loss))
        
        score_file.write("Epoch %d, LR %f, Train Loss %f, Train ACC %2.2f%%, Val Loss %f, Val ACC %2.2f%%\n"
                         %(epoch, lr, loss, acc, val_loss, val_acc))
        score_file.flush()

        # --- BEST MODEL & EARLY STOPPING MANTIĞI ---
        if val_loss < best_val_loss:
            print(f" -> Yeni rekor! Val Loss {best_val_loss:.4f} -> {val_loss:.4f}. 'best_model.model' kaydediliyor.")
            best_val_loss = val_loss
            s.save_parameters(args.save_path + "/best_model.model")
            patience_counter = 0 # Sayacı sıfırla
        else:
            patience_counter += 1
            print(f" -> İyileşme yok. Sabır: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(" -> Erken Durdurma (Early Stopping) tetiklendi. Eğitim bitiriliyor.")
                break # Döngüyü kır
        # ---------------------------------------------

    if epoch >= args.max_epoch:
        break

    epoch += 1