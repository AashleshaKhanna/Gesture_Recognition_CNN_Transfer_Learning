"""Evaluate an ASL gesture recognition checkpoint."""
from __future__ import annotations
import argparse, torch
from src.data import make_loaders
from src.models import build_model
from src.utils import evaluate_model

def main():
    p=argparse.ArgumentParser(description='Evaluate ASL classifier checkpoint.')
    p.add_argument('--data-dir',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--model',default='custom')
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--channels',type=int,nargs=4,default=[32,64,128,256]); p.add_argument('--dropout',type=float,default=0.5); p.add_argument('--device',default=None)
    args=p.parse_args(); device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    split=make_loaders(args.data_dir,batch_size=args.batch_size,image_size=224)
    model=build_model(args.model, len(split.class_names), tuple(args.channels), args.dropout)
    ckpt=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ckpt.get('model_state',ckpt))
    print(f'Test accuracy: {evaluate_model(model, split.test_loader, device):.4f}')
if __name__=='__main__': main()
