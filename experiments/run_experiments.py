"""Run hyperparameter experiments for the custom ASL CNN."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from src.data import make_loaders
from src.models import ConfigurableASLCNN
from src.utils import save_history_csv, train_model

def main():
    p=argparse.ArgumentParser(description='Run ASL CNN hyperparameter experiments.'); p.add_argument('--data-dir',required=True); p.add_argument('--epochs',type=int,default=10); p.add_argument('--device',default=None); args=p.parse_args()
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    experiments=[
        {'name':'E1_base','channels':(32,64,128,256),'batch_size':32,'learning_rate':1e-3,'weight_decay':1e-4},
        {'name':'E2_lr_low','channels':(32,64,128,256),'batch_size':32,'learning_rate':3e-4,'weight_decay':1e-4},
        {'name':'E3_batch_big','channels':(32,64,128,256),'batch_size':64,'learning_rate':1e-3,'weight_decay':1e-4},
        {'name':'E4_more_capacity','channels':(64,128,256,256),'batch_size':32,'learning_rate':1e-3,'weight_decay':1e-4},]
    Path('results').mkdir(exist_ok=True); Path('checkpoints').mkdir(exist_ok=True)
    for exp in experiments:
        print('='*80); print(f"Running {exp['name']}: {exp}"); print('='*80)
        split=make_loaders(args.data_dir,batch_size=exp['batch_size'])
        model=ConfigurableASLCNN(num_classes=len(split.class_names), channels=exp['channels'], dropout=0.5)
        result=train_model(model,split.train_loader,split.val_loader,args.epochs,exp['learning_rate'],exp['weight_decay'],f"checkpoints/{exp['name']}.pt",device)
        save_history_csv(result['history'],f"results/{exp['name']}_history.csv")
        print(f"{exp['name']} best validation accuracy: {result['best_val_acc']:.4f}")
if __name__=='__main__': main()
