"""Train the custom ASL CNN."""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.data import make_loaders, make_small_subset_loader
from src.models import build_model
from src.utils import ensure_dirs, save_history_csv, train_model

def overfit_small_dataset(model, loader, iterations:int=200, learning_rate:float=1e-3, device=None):
    dev=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu')); model=model.to(dev).train()
    criterion=nn.CrossEntropyLoss(); optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    x,y=next(iter(loader)); x,y=x.to(dev),y.to(dev)
    for i in range(1,iterations+1):
        optimizer.zero_grad(set_to_none=True); logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
        acc=(logits.argmax(dim=1)==y).float().mean().item()
        if i==1 or i%10==0: print(f'Iteration {i:03d} | loss {loss.item():.4f} | acc {acc:.2%}')
        if acc==1.0: print(f'Reached 100% training accuracy at iteration {i}.'); break

def main():
    p=argparse.ArgumentParser(description='Train ASL gesture recognition CNN.')
    p.add_argument('--data-dir',required=True); p.add_argument('--model',default='custom'); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=3e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--channels',type=int,nargs=4,default=[32,64,128,256]); p.add_argument('--dropout',type=float,default=0.5); p.add_argument('--checkpoint',default='checkpoints/best_custom.pt')
    p.add_argument('--small-overfit',action='store_true'); p.add_argument('--device',default=None)
    args=p.parse_args(); ensure_dirs(); device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    split=make_loaders(args.data_dir,batch_size=args.batch_size,image_size=224,imagenet_norm=False)
    model=build_model(args.model, num_classes=len(split.class_names), channels=tuple(args.channels), dropout=args.dropout)
    if args.small_overfit:
        overfit_small_dataset(model, make_small_subset_loader(split,64,64), max(args.epochs,200), args.learning_rate, str(device)); return
    result=train_model(model, split.train_loader, split.val_loader, args.epochs, args.learning_rate, args.weight_decay, args.checkpoint, device)
    save_history_csv(result['history'],'results/custom_cnn_history.csv')
    print(f"Best validation accuracy: {result['best_val_acc']:.4f}"); print(f"Best checkpoint saved to: {result['checkpoint_path']}")
if __name__=='__main__': main()
