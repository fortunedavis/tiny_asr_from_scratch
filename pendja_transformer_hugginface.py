from transformers import TrainingArguments, Trainer
from dataclasses import dataclass 
from typing import Optional 
import torch 
from torch.nn.utils.rnn import pad_sequence 
from transformers.utils import ModelOutput 
import json
import torch.optim as optim 
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torchaudio 
import evaluate 
import unicodedata
import logging 
import datetime
import math 
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pendja_transformer import BestDataset, myvocab, ctc_decode, BestASR



wer_compute =  evaluate.load("wer")
cer_compute =  evaluate.load("cer")
loss_fn = nn.CTCLoss(blank=0,zero_infinity=True)  
# optimizer = optim.AdamW(model.parameters(),lr=1e-4, weight_decay=0.01)

train_path = "/home/fortu/Documents/manip_dataset/asr_train.csv"
valid_path = "/home/fortu/Documents/manip_dataset/asr_valid.csv"
test_path = "/home/fortu/Documents/manip_dataset/asr_test.csv"
vocab_file ="/home/fortu/Documents/Preparations/vocab.json"
root ="/home/fortu/Documents/manip_dataset/fongbe_one"

train_data = BestDataset(train_path,root,vocab_file)
test_data = BestDataset(test_path,root,vocab_file)
valid_data = BestDataset(valid_path,root,vocab_file)

vocab = myvocab(vocab_file)
id2label = {id:label for label, id in vocab.items()}
    
@dataclass 
class CTCOutput:
    loss : Optional[torch.tensor] = None
    logits : Optional[torch.tensor] = None
    
class PendjaTransformerCTC(nn.Module):
    def __init__(self,asr_model,loss_fn):
        super().__init__()
        self.model = asr_model 
        self.loss_fn = loss_fn 
        
    def forward(
        self,
        audio,
        audio_length,
        label_token = None,
        token_lengths = None
    ):
      log_probs = self.model(audio)
    #   print(f"Shape log_probs_ctc (T, B, C): {log_probs.shape} | Max target length: {label_token.shape[1]}")
      loss = None 
      if label_token is not None and token_lengths is not None:
          loss = self.loss_fn(log_probs, label_token, audio_length, token_lengths)
      return {"loss": loss, "logits" : log_probs.permute(1, 0, 2)}        
  
def collate_fn(batch):
    # print(batch)
    audios = [b["audio"].transpose(1,0) for b in batch] # T,C, B
    pad_audios = pad_sequence(audios, batch_first=True).transpose(1,2) # B, T, C
    
    labels = pad_sequence(
        [b["label_token"] for b in batch],
        batch_first=True,
        padding_value=-100
        )
    return {
        "audio": pad_audios,                          # ← matche forward(audios=...)
        "audio_length": torch.tensor([b["audio_length"] for b in batch]),
        "label_token": labels,
        "token_lengths": torch.tensor([len(b["label_token"]) for b in batch], dtype=torch.long),
    }  


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
     


training_args = TrainingArguments(
    output_dir="./pendja_checkpoints",
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    warmup_ratio=0.1,                    
    lr_scheduler_type="linear",         
    weight_decay=0.02,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
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
    # print(train_data[0])

