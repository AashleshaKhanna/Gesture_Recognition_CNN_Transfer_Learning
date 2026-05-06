"""Shared training and evaluation utilities."""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn

def ensure_dirs():
    Path('checkpoints').mkdir(exist_ok=True); Path('results').mkdir(exist_ok=True)

def accuracy_from_logits(logits:torch.Tensor, y:torch.Tensor)->float:
    return (logits.argmax(dim=1)==y).float().mean().item()

def run_one_epoch(model:nn.Module, loader, criterion, optimizer=None, device='cpu'):
    is_train=optimizer is not None; model.train(is_train)
    total_loss=total_acc=n=0
    for x,y in loader:
        x,y=x.to(device), y.to(device)
        if is_train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits=model(x); loss=criterion(logits,y)
            if is_train: loss.backward(); optimizer.step()
        total_loss+=loss.item(); total_acc+=accuracy_from_logits(logits,y); n+=1
    return total_loss/n, total_acc/n

def train_model(model:nn.Module, train_loader, val_loader, epochs:int=10, learning_rate:float=3e-4, weight_decay:float=1e-4, checkpoint_path:str='checkpoints/best_model.pt', device='cpu')->dict[str,Any]:
    ensure_dirs(); model=model.to(device)
    criterion=nn.CrossEntropyLoss(); optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    history={'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
    best_val_acc=0.0; best_state=None
    for epoch in range(1,epochs+1):
        tr_loss,tr_acc=run_one_epoch(model,train_loader,criterion,optimizer,device)
        va_loss,va_acc=run_one_epoch(model,val_loader,criterion,None,device)
        for k,v in [('train_loss',tr_loss),('train_acc',tr_acc),('val_loss',va_loss),('val_acc',va_acc)]: history[k].append(v)
        print(f'Epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}')
        if va_acc>best_val_acc:
            best_val_acc=va_acc; best_state=copy.deepcopy(model.state_dict())
            torch.save({'epoch':epoch,'model_state':best_state,'best_val_acc':best_val_acc,'history':history}, checkpoint_path)
    if best_state is not None: model.load_state_dict(best_state)
    return {'model':model,'history':history,'best_val_acc':best_val_acc,'checkpoint_path':checkpoint_path}

@torch.no_grad()
def evaluate_model(model:nn.Module, loader, device='cpu')->float:
    model.eval(); model.to(device); correct=total=0
    for x,y in loader:
        x,y=x.to(device), y.to(device); preds=model(x).argmax(dim=1)
        correct += int((preds==y).sum().item()); total += y.size(0)
    return correct/total

def save_history_csv(history:dict[str,list[float]], path:str):
    keys=['train_loss','train_acc','val_loss','val_acc']; arr=np.column_stack([history[k] for k in keys])
    np.savetxt(path, arr, delimiter=',', header=','.join(keys), comments='')
