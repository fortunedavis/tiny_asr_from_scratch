import torch 
import torchaudio 
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import json 
# import torchmetrics 
import evaluate 
import unicodedata
import logging 
import datetime
import math 

device = "cuda" if torch.cuda.is_available() else "cpu"

log_filename = f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BestDataset(Dataset):
    def __init__(self,csv_path, root, vocab_file):
        self.csv_path = csv_path
        self.data =  pd.read_csv(self.csv_path)
        self.data["message"] =  self.data["message"].fillna("").apply(remove_tone)
        self.vocab = myvocab(vocab_file)
        self.root = root
        self.conv_dims = [
                (1,512,5,5),
                (512,512,5,5),
                (512,512,5,3),
                (512,512,5,2),
                (512,512,5,2),
                (512,512,5,2),
                (512,512,5,2)
            ]
        self.data =  self._filter_invalid_samples()
        
    def __getitem__(self, key):
        audio_path = Path(self.root,str(self.data.iloc[key]["ID"])).with_suffix(".wav")
        if Path.is_file(audio_path):
            wav, sr =  torchaudio.load(audio_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16_000:
                wav = torchaudio.transforms.Resample(orig_freq=sr,new_freq=16_000)(wav)
            wav_length = self.conv_out_length(wav.shape[1],self.conv_dims)
            label = self.data.iloc[key]["message"]
            tokens = torch.tensor([self.vocab.get(el, self.vocab["[UNK]"]) for el in label], dtype=torch.long)
            token_length = len(tokens)
        return {
            "audio" : wav,
            "audio_length" : wav_length,
            "label_token" : tokens,
            "token_lengths" : token_length,
            # "label_text":label
        }
    
    def __len__(self):
        return len(self.data)
    
    def _filter_invalid_samples(self):
        
        data_rows = []
        for idx in range(len(self.data)):
            audio_path = Path(self.root,str(self.data.iloc[idx]["ID"])).with_suffix(".wav")
            if not audio_path.is_file():
                continue
            try:
                info =  torchaudio.info(audio_path)
                n_samples = info.num_frames
                if info.sample_rate !=16000:
                    n_samples = int(16000*info.num_frames / info.sample_rate)
                    
                audio_length = self.conv_out_length(n_samples,self.conv_dims)
            except Exception:
                continue
            
            label = self.data.iloc[idx]["message"]
            tokens = [self.vocab.get(c, self.vocab["[UNK]"]) for c in label]
            
            min_need = len(tokens)
            
            for i in range(1,len(tokens)):
                if tokens[i] == tokens[i-1]:
                    min_need+=1
            
            if audio_length >= min_need and len(tokens) > 0:
                data_rows.append((idx,n_samples))
                
        data_sorted = sorted(data_rows, key= lambda x: x[1])
        idx_sorted = [idx for idx,_ in data_sorted]
        
        logger.info(f"Filtrage : {len(data_rows)}/{len(self.data)} exemples valides")

        return self.data.iloc[idx_sorted].reset_index(drop=True)
        
    def conv_out_length(self, lengths, conv_dims, padding=None, dilation=None):
        if padding is None:
            padding = len(conv_dims) * [0]
        if dilation is None:
            dilation = len(conv_dims) * [1]
        for (_, _, kernel, stride), pad, dil in zip(conv_dims, padding, dilation):
            lengths = ((lengths + 2*pad - dil*(kernel-1) - 1) // stride) + 1
        return lengths

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x : (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class BestASR(nn.Module):
    def __init__(self,  vocab_size=32, d_model=512, n_head=8, num_layers=6):
        super().__init__()
       #use log mel here instead
        self.conv_dims = [
                (1,512,5,5),
                (512,512,5,5),
                (512,512,5,3),
                (512,512,5,2),
                (512,512,5,2),
                (512,512,5,2),
                (512,512,5,2)
            ]
        # self.features = nn.Sequential(nn.Conv1d(*dim) for dim in self.conv_dims)
        self.features_extractor = nn.Sequential(
                *[
                    nn.Conv1d(*dim)
                    for dim in self.conv_dims
                ]
            )  #B,C,T
        
        self.gelu = nn.GELU()
        
        self.dropout = nn.Dropout(0.3)
        
        self.norm = nn.BatchNorm1d(128)
        
        self.cnn_proj = nn.Linear(d_model, d_model)
        
        self.pos_enc = PositionalEncoding(d_model, dropout=0.3)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model*2,
            activation='gelu',
            batch_first=True,
            dropout=0.3
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers   
        )
        

        self.proj = nn.Linear(d_model,vocab_size)

        self.gnorm = nn.GroupNorm(num_groups=8, num_channels=512)
        
    def forward(self,x):
        
        feats = self.features_extractor(x) #B,C,T

        feats =  self.gelu(self.gnorm(feats))

        feats = feats.permute(0,2,1)
        
        feat_proj = self.cnn_proj(feats) #B T C
        
        # feat_proj = self.pos_enc(feat_proj) 
        
        enc_output = self.encoder(feat_proj)
        
        x = self.dropout(enc_output)
        
        logits = self.proj(x) # B,T,C
        
        logs_probs = F.log_softmax(logits, dim=-1)
    
        logs_probs = logs_probs.permute(1,0,2) #T B C
        
        return logs_probs

def get_scheduler_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # montée linéaire
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))  # cosine après
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, train_loader, valid_loader, loss_fn, id2label, scheduler,optimizer):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader,desc="Training")
    for batch in loop:
        inputs = batch["audios"]
        input_length = batch["audio_length"]
        label_tokens = batch["label_tokens"]
        label_length = batch["token_lengths"]
        label_texts = batch["label_texts"]
        
        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        
        log_probs = model(inputs)
        
        # print("contient Inf:", torch.isnan(log_probs).any().item())
        # print("Shape réelle T:", log_probs.shape[0])
        # print("input_length déclaré:", input_length)
        # print(f"{log_probs.shape}")
        # print(label_tokens)
        # print(f"La taille de l'input {input_length}")
        # print(label_length)
        
        def ctc_min_input_length(tokens):
            min_len = len(tokens)
            for i in range(1, len(tokens)):
                if tokens[i] == tokens[i-1]:
                    min_len += 1
            return min_len

        # Utilise les vraies valeurs du batch qui plante
        splits = label_tokens.split(label_length.tolist())

        # for i, tok in enumerate(splits):
        #     tok_list = tok.tolist()
        #     needed = ctc_min_input_length(tok_list)
        #     logger.info(f"Exemple {i}: input_length={input_length[i].item()}, label_length={len(tok_list)}, min_CTC_requis={needed}")
        #     if needed > input_length[i].item():
        #         logger.info(f"IMPOSSIBLE pour CTC — manque {needed - input_length[i].item()} frames")
        #     else:
        #         logger.info(f"OK — marge de {input_length[i].item() - needed} frames")
                
        
        loss =  loss_fn(log_probs, label_tokens, input_length, label_length)
        # print("input_length:", input_length)
        # print("label_length:", label_length)
        # print("label toke:", label_tokens)
        # print("contient Inf:", torch.isnan(loss).any().item())
        # print(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    logger.info(f"train Loss {total_loss/len(train_loader):.4f}")
    
    valid_loss = 0
    all_output = []
    labels_list = []
    loop_val = tqdm(valid_loader,desc="valid")
    model.eval()
    with torch.no_grad():
        for batch in loop_val:
            inputs = batch["audios"]
            input_length = batch["audio_length"]
            label_tokens = batch["label_tokens"]
            label_length = batch["token_lengths"]
            label_texts = batch["label_texts"]
            
            inputs = inputs.to(device)
            
            log_probs = model(inputs)
            loss =  loss_fn(log_probs, label_tokens, input_length, label_length)
            
            valid_loss += loss.item()
            
            # print(valid_loss)
            
            pred_ids = torch.argmax(log_probs,dim=-1) #T B
            
            pred_ids = pred_ids.transpose(1,0) #B T 
            for i in range(pred_ids.size(0)):
                decoded = ctc_decode(pred_ids[i].tolist(), blank_id=0)
                text = "".join(id2label[d] for d in decoded)
                all_output.append(text)
            # print(all_output)
            labels_list.extend(label_texts)
            print(labels_list)
            print(all_output)
            
            # print(labels_list)
            # print(all_output)     
            # print(labels_list)
            # print(len(all_output)) 
            # print(len(labels_list))
            
        wer = wer_compute.compute(predictions = all_output, references = labels_list)
        cer = cer_compute.compute(predictions = all_output, references = labels_list)
        
        logger.info(f"Valid Loss {valid_loss/len(valid_loader):.4f}")
        logger.info(f"Valid WER {100*wer:.4f} | Valid CER {100*cer:.4f}")
        
def test(model,loader,id2lab):
    model.eval()
    with torch.no_grad():
        all_output = []
        labels_list = []
        
        loop_test = tqdm(loader,desc="Test")
        for batch in loop_test:
            inputs = batch["audios"]
            label_texts = batch["label_texts"]
            
            inputs = inputs.to(device)
            
            log_probs = model(inputs)
            pred_ids = torch.argmax(log_probs,dim=-1) #T B
            pred_ids = pred_ids.transpose(1,0) #B T 
            
            for i in range(pred_ids.size(0)):
                decoded = ctc_decode(pred_ids[i].tolist(), blank_id=0)
                text = "".join(id2lab[d] for d in decoded)
                all_output.append(text)
            # print(all_output)
            labels_list.extend(label_texts)
            # print(labels_list)
            
        wer = wer_compute.compute(predictions = all_output, references = labels_list)
        cer = cer_compute.compute(predictions = all_output, references = labels_list)
        logger.info(f"Test WER {100*wer:.4f} | Test CER {100*cer:.4f}")
    

def myvocab(vocab_file):
    
    with open(vocab_file, encoding="utf-8") as f:
        vocab =  json.load(f)
    vocab = {token: idx+1 for token,idx in vocab.items()} 
    vocab["[BLANK]"] = 0

    if "[UNK]" not in vocab:
        vocab["[UNK]"] = len(vocab)
    logger.info(f"vocab real size is {len(vocab)}")
    assert len(vocab) == 32, "vocab size mismatch"  
    
    vocab = dict(sorted(vocab.items(), key = lambda item:item[1]))
    return vocab


def collate_fn(batch):
    
    audios = [b["audio"].transpose(0,1) for b in batch]
    
    audio_lengths = torch.tensor([b["audio_length"] for b in batch])
    
    # label_tokens = torch.tensor([b["label_token"] for b in batch])
    
    # token_lengths = torch.tensor([b["token_lengths"] for b in batch])
    
    label_texts = [b["label_text"] for b in batch]
    
    pad_audios = pad_sequence(
        audios, batch_first=True
    )
    
    pad_audios = pad_audios.transpose(1, 2) 
    
    label_tokens = torch.cat([b["label_token"] for b in batch])
    
    token_lengths = torch.tensor([len(b["label_token"]) for b in batch], dtype=torch.long)

    
    return {
            "audios" : pad_audios ,
            "audio_length" : audio_lengths,
            "label_tokens" : label_tokens,
            "token_lengths" : token_lengths,
            "label_texts" : label_texts
        }


def ctc_decode(tokens, blank_id=0):
    decoded = [] 
    prev = None
    for t in tokens:
        if t!= prev and t != blank_id:
            decoded.append(t)
        prev = t 
    return decoded
 
def remove_tone(text):
    text =  str(text)
    text =  "".join(c for c in unicodedata.normalize("NFD",text)
                    if unicodedata.category(c) != "Mn")
    replacements = {
        #  "ɔ": "o",
        # "ɛ": "e",
        "đ" : "d",
        "ɖ": "d",
        "ð": "d",
        "œ": "oe",
        "ε": "ɛ",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    for c in ["‚", "’", "‘", "“", "”"]:
        text = text.replace(c, "")
        
    text = "".join(
        c for c in str(text)
        if c not in ["\u200b", "\u200c", "\u200d"]
    )
    
    text = text.replace("-", " ")
    return text

if __name__ == "__main__":
    
    model =  BestASR()   
    
    params =  sum(el.numel() for el in model.parameters())
    
    print(f"the number of params is {params}")
    print(model)
    logger.info(f"the number of params is {params}")
    
    model = model.to(device)
    loss_fn = nn.CTCLoss(blank=0,zero_infinity=True)  
    optimizer = optim.AdamW(model.parameters(),lr=1e-4, weight_decay=0.01)
    
    train_path = "/home/fortu/Documents/manip_dataset/asr_train.csv"
    valid_path = "/home/fortu/Documents/manip_dataset/asr_valid.csv"
    test_path = "/home/fortu/Documents/manip_dataset/asr_test.csv"
    vocab_file ="/home/fortu/Documents/Preparations/vocab.json"
    root ="/home/fortu/Documents/manip_dataset/fongbe_one"

    vocab = myvocab(vocab_file)
    id2lab = {id:label for label, id in vocab.items()}

    train_data = BestDataset(train_path,root,vocab_file)
    test_data = BestDataset(test_path,root,vocab_file)
    valid_data = BestDataset(valid_path,root,vocab_file)

    train_loader = DataLoader(train_data, batch_size=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=2, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=2, collate_fn=collate_fn)


    wer_compute =  evaluate.load("wer")
    cer_compute =  evaluate.load("cer")

    # item =  next(iter(train_loader))
    # item =  next(iter(train_loader))
    # print(item)

# hyperparameters
    epochs = 10

    total_steps = epochs * len(train_loader)
    warmup_steps = total_steps // 10  # 10% des steps en warmup

    scheduler = get_scheduler_with_warmup(optimizer, warmup_steps, total_steps)
    
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, valid_loader, loss_fn, id2lab, scheduler, optimizer)
        logger.info(f"LR actuel : {scheduler.get_last_lr()[0]:.6f}")
        torch.save(
            {"epoch" : epoch,
            "model_state_dict":model.state_dict(),
            }, f"Pendja_{epoch+1}.pt"
        )
        logger.info("-------------Training done------------------")
        
    test(model,test_loader,id2lab)
