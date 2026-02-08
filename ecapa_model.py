'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, time, numpy, tqdm
import torch.nn as nn
from loss import AAMsoftmax
from model import ECAPA_TDNN

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        
        # ECAPA-TDNN Modelini Başlat (C kanalı ile)
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
        
        # Loss Fonksiyonu (Classifier)
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        
        print(time.strftime("%m-%d %H:%M:%S") + " Model Parameter Count = %.2f M"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        # Update the learning rate based on the current epoch
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            
            # Forward pass (Augmentasyon dataloader'da kapalı ama model içi SpecAugment açık kalabilir)
            # Eğer model içi augmentation da istemiyorsan aug=False yap.
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True) 
            
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)          
            nloss.backward()
            self.optim.step()
            
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / len(loader))) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
            sys.stderr.flush()
        
        sys.stdout.write("\n")
        return loss/num, lr, top1/index*len(labels)

    def eval_network(self, loader):
        """
        Modified for Validation Accuracy instead of EER pair verification.
        Evaluates the model on the validation set (classification task).
        """
        self.eval()
        loss, top1, index = 0, 0, 0
        
        with torch.no_grad():
            for num, (data, labels) in tqdm.tqdm(enumerate(loader, start=1), total=len(loader)):
                labels = torch.LongTensor(labels).cuda()
                
                # Validation'da Augmentation KAPALI olmalı
                speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=False)
                
                # Loss ve Accuracy hesapla
                nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
                
                loss += nloss.detach().cpu().numpy()
                top1 += prec
                index += len(labels)

        avg_loss = loss / num
        avg_acc  = top1 / index * len(labels) # AAMSoftmax'tan dönen prec ortalaması
        
        return avg_loss, avg_acc

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)