"""Plot training curves from a saved CSV history file."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_history(csv_path:str, output_prefix:str='results/training'):
    data=np.genfromtxt(csv_path,delimiter=',',names=True); epochs=np.arange(1,len(data['train_loss'])+1)
    plt.figure(figsize=(8,5)); plt.plot(epochs,data['train_loss'],label='train_loss'); plt.plot(epochs,data['val_loss'],label='val_loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training vs Validation Loss'); plt.legend(); Path(output_prefix).parent.mkdir(parents=True,exist_ok=True); plt.savefig(f'{output_prefix}_loss.png',bbox_inches='tight'); plt.close()
    plt.figure(figsize=(8,5)); plt.plot(epochs,data['train_acc'],label='train_acc'); plt.plot(epochs,data['val_acc'],label='val_acc'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy'); plt.legend(); plt.savefig(f'{output_prefix}_accuracy.png',bbox_inches='tight'); plt.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',default='results/custom_cnn_history.csv'); p.add_argument('--output-prefix',default='results/custom_cnn'); args=p.parse_args(); plot_history(args.csv,args.output_prefix)
if __name__=='__main__': main()
