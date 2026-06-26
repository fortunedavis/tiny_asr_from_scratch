import torch 
import torchaudio 
import torchaudio.transforms as T
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
import evaluate 
import unicodedata
import logging 
import datetime
import math 
from transformers import TrainingArguments, Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"


wer_compute =  evaluate.load("wer")
cer_compute =  evaluate.load("cer")
loss_fn = nn.CTCLoss(blank=0,zero_infinity=True)  

train_path = "/home/fortu/Documents/manip_dataset/asr_train.csv"
valid_path = "/home/fortu/Documents/manip_dataset/asr_valid.csv"
test_path = "/home/fortu/Documents/manip_dataset/asr_test.csv"
vocab_file ="/home/fortu/Documents/Preparations/tiny_asr_from_scratch/vocab.json"
root ="/home/fortu/Documents/manip_dataset/fongbe_one"





class BestDataset(Dataset):
    def __init__(self,csv_path, root, vocab_file):
        self.csv_path = csv_path
        self.data =  pd.read_csv(self.csv_path)
        self.data["message"] =  self.data["message"].fillna("").apply(remove_tone)
        self.vocab = myvocab(vocab_file)
        self.root = root
        self.conv_dims = [
                (80,128,5,2),
                (128,32,5,2),
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
            
            mel = compute_log_mel(wav)
            
            wav_length = self.conv_out_length(mel.shape[2],self.conv_dims)
            
            #ici
            
            label = self.data.iloc[key]["message"]
            tokens = torch.tensor([self.vocab.get(el, self.vocab["[UNK]"]) for el in label], dtype=torch.long)
            token_length = len(tokens)
        
        return {
            "audio" : mel,
            "audio_length" : wav_length,
            "label_token" : tokens,
            "token_lengths" : token_length,
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
                n_mel_frames = (n_samples - 1024) // 512 + 1
                audio_length = self.conv_out_length(n_mel_frames,self.conv_dims)
                
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
        

        return self.data.iloc[idx_sorted].reset_index(drop=True)
        
    def conv_out_length(self, lengths, conv_dims, padding=None, dilation=None):
        if padding is None:
            padding = len(conv_dims) * [0]
        if dilation is None:
            dilation = len(conv_dims) * [1]
        for (_, _, kernel, stride), pad, dil in zip(conv_dims, padding, dilation):
            lengths = ((lengths + 2*pad - dil*(kernel-1) - 1) // stride) + 1
        return lengths

def compute_log_mel(wav, 
                    sample_rate=16_000, 
                    n_fft=1024, 
                    win_length=1024, 
                    hop_length=512,
                    n_mels=80):
    
    mel_log =  T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels
    )(wav)
    
    ampli_db = T.AmplitudeToDB(top_db=80, stype="power")(mel_log)
    
    return ampli_db

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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class BestASR(nn.Module):
    def __init__(self,  vocab_size=32, d_model=512, n_head=8, num_layers=6):
        super().__init__()
        self.conv_dims =  self.conv_dims = [
                (80,128,5,2),
                (128,32,5,2),
            ]
        
        self.features_extractor = nn.Sequential(
                *[
                    nn.Conv1d(*dim)
                    for dim in self.conv_dims
                ]
            )  #B,C,T
        
        self.gelu = nn.GELU()
        
        self.dropout = nn.Dropout(0.3)
        
        self.norm = nn.BatchNorm1d(128)
        
        self.cnn_proj = nn.Linear(32, d_model)
        
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

        self.gnorm = nn.GroupNorm(num_groups=4, num_channels=32)
        
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

    

def myvocab(vocab_file):
    
    with open(vocab_file, encoding="utf-8") as f:
        vocab =  json.load(f)
    vocab = {token: idx+1 for token,idx in vocab.items()} 
    vocab["[BLANK]"] = 0

    if "[UNK]" not in vocab:
        vocab["[UNK]"] = len(vocab)
    assert len(vocab) == 32, "vocab size mismatch"  
    
    vocab = dict(sorted(vocab.items(), key = lambda item:item[1]))
    return vocab

def collate_fn(batch):
    
    audio_lengths = torch.tensor([b["audio_length"] for b in batch])
    
    # pad_audios = pad_sequence(
    #     [b["audio"].squeeze(0).T for b in batch], batch_first=True
    # )
    # print([b["audio"].shape for b in batch])
    
    audios = [b["audio"].squeeze(0).permute(1,0)  for b in batch] # T,C, B
    pad_audios = pad_sequence(audios, batch_first=True).transpose(1,2) # B, T, C
    
    labels = pad_sequence(
        [b["label_token"] for b in batch],
        batch_first=True,
        padding_value=-100
        )
    
    token_lengths = torch.tensor([len(b["label_token"]) for b in batch], dtype=torch.long)

    return {
            "audios" : pad_audios ,
            "audio_length" : audio_lengths,
            "label_tokens" : labels,
            "token_lengths" : token_lengths,
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


def compute_metrics(eval_preds):
  
    log_probs, labels = eval_preds

    log_probs = torch.tensor(log_probs)
    
    pred_ids = torch.argmax(log_probs,dim=-1)
    
    all_output = []
    
    for i in range(pred_ids.shape[0]):
        decoded = ctc_decode(pred_ids[i].tolist(), blank_id=0)
        all_output.append("".join(id2label[d] for d in decoded))
        
    references = []
    for label_seq in labels:
        label_seq = [x for x in label_seq if x != -100]
        references.append(
            "".join(id2label[x] for x in label_seq)
        ) 
    
    wer = wer_compute.compute(predictions=all_output,references=references)
    cer = cer_compute.compute(predictions=all_output,references=references)
    return {"wer": wer, "cer": cer}

class PendjaTransformerCTC(nn.Module):
    def __init__(self,asr_model,loss_fn):
        super().__init__()
        self.model = asr_model 
        self.loss_fn = loss_fn 
        
    def forward(
        self,
        audios,
        audio_length,
        label_tokens = None,
        token_lengths = None
    ):
      log_probs = self.model(audios)
    #   print(f"Shape log_probs_ctc (T, B, C): {log_probs.shape} | Max target length: {label_token.shape[1]}")
      loss = None 
      if label_tokens is not None and token_lengths is not None:
          loss = self.loss_fn(log_probs, label_tokens, audio_length, token_lengths)
      return {"loss": loss, "logits" : log_probs.permute(1, 0, 2)}   
  
train_data = BestDataset(train_path,root,vocab_file)
test_data = BestDataset(test_path,root,vocab_file)
valid_data = BestDataset(valid_path,root,vocab_file)

training_args = TrainingArguments(
    output_dir="./pendja_checkpoints",
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    # warmup_ratio=0.1,                    
    lr_scheduler_type="linear",         
    weight_decay=0.02,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),      
    dataloader_num_workers=2,
    report_to="none", 
    remove_unused_columns=False, 
    max_grad_norm=0.5,         
)

device = "cuda" if torch.cuda.is_available() else "cpu"

vocab = myvocab(vocab_file)
id2label = {id:label for label, id in vocab.items()}

base_model = BestASR().to(device)

loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

wrapped_model = PendjaTransformerCTC(base_model, loss_fn)

trainer = Trainer(
    model=wrapped_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)


if __name__ == "__main__":
    trainer.train()



  