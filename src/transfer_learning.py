"""AlexNet feature extraction and transfer-learning classifier training."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from src.data import make_loaders
from src.models import AlexNetFeatureClassifier
from src.utils import evaluate_model, save_history_csv, train_model

@torch.no_grad()
def extract_features_from_loader(alexnet_features:nn.Module, loader:DataLoader, device):
    alexnet_features.eval(); feats=[]; labels=[]
    for x,y in loader:
        feats.append(alexnet_features(x.to(device)).cpu()); labels.append(y)
    return torch.cat(feats,0), torch.cat(labels,0)

def extract_alexnet_features(data_dir:str, output_dir:str, batch_size:int=64, device=None):
    dev=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    split=make_loaders(data_dir,batch_size=batch_size,image_size=224,imagenet_norm=True)
    alexnet=models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(dev).eval()
    for p in alexnet.parameters(): p.requires_grad=False
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    for name,loader in [('train',split.train_loader),('val',split.val_loader),('test',split.test_loader)]:
        x,y=extract_features_from_loader(alexnet.features,loader,dev); torch.save((x,y),out/f'alex_{name}.pt'); print(name, tuple(x.shape))

def train_transfer_classifier(feature_dir:str, epochs:int=10, batch_size:int=32, learning_rate:float=3e-4, weight_decay:float=1e-4, device=None):
    dev=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu')); fp=Path(feature_dir)
    xtr,ytr=torch.load(fp/'alex_train.pt',map_location='cpu'); xv,yv=torch.load(fp/'alex_val.pt',map_location='cpu'); xt,yt=torch.load(fp/'alex_test.pt',map_location='cpu')
    train_loader=DataLoader(TensorDataset(xtr,ytr),batch_size=batch_size,shuffle=True); val_loader=DataLoader(TensorDataset(xv,yv),batch_size=batch_size); test_loader=DataLoader(TensorDataset(xt,yt),batch_size=batch_size)
    model=AlexNetFeatureClassifier(num_classes=int(torch.max(ytr).item()+1))
    result=train_model(model,train_loader,val_loader,epochs,learning_rate,weight_decay,'checkpoints/best_transfer_alexnet.pt',dev)
    save_history_csv(result['history'],'results/transfer_alexnet_history.csv')
    print(f"Best transfer validation accuracy: {result['best_val_acc']:.4f}"); print(f"Transfer-learning test accuracy: {evaluate_model(result['model'],test_loader,dev):.4f}")

def main():
    p=argparse.ArgumentParser(description='AlexNet transfer learning workflow.'); sp=p.add_subparsers(dest='command',required=True)
    e=sp.add_parser('extract'); e.add_argument('--data-dir',required=True); e.add_argument('--output-dir',default='results/alexnet_features'); e.add_argument('--batch-size',type=int,default=64); e.add_argument('--device',default=None)
    t=sp.add_parser('train'); t.add_argument('--feature-dir',default='results/alexnet_features'); t.add_argument('--epochs',type=int,default=10); t.add_argument('--batch-size',type=int,default=32); t.add_argument('--learning-rate',type=float,default=3e-4); t.add_argument('--weight-decay',type=float,default=1e-4); t.add_argument('--device',default=None)
    args=p.parse_args()
    if args.command=='extract': extract_alexnet_features(args.data_dir,args.output_dir,args.batch_size,args.device)
    else: train_transfer_classifier(args.feature_dir,args.epochs,args.batch_size,args.learning_rate,args.weight_decay,args.device)
if __name__=='__main__': main()
