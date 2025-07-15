# This code currently only for direct inference 24 class
import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from util.datasets import build_chr_dataset
import argparse
from engine_finetune import evaluate
import torch.nn.functional as F


parser = argparse.ArgumentParser('Fine-tuning for image classification')

parser.add_argument('--model', type=str, default='vit_large_patch16', help='Model architecture')
parser.add_argument('--input_size', type=int, default=224, help='Input size for training')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
parser.add_argument('--nb_classes', type=int, default=24, help='')
parser.add_argument('--drop_path', type=float, default=0.2, help='')
parser.add_argument('--global_pool', type=str, default='avg', help='')
parser.add_argument('--w_pad', type=bool, default=True, help='')
parser.add_argument('--checkpoint_path', type=str, default='./finetune_chr_24_class/checkpoint-best.pth', help='')
parser.add_argument('--data_path', default='/ibex/project/c2277/data/Karyotype/class_abnormal/pretrain/all_data_2', type=str,
                        help='dataset path')
parser.add_argument('--task', default='./finetune_chr_24_class/', type=str, help='task name')
parser.add_argument('--output_dir', type=str, default='./finetune_chr_24_class', help='Output directory')

args = parser.parse_args()

model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
checkpoint_model = checkpoint['model']
# state_dict = model.state_dict()
# for k in ['head.weight', 'head.bias']:
#     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#         print(f"Removing key {k} from pretrained checkpoint")
#         del checkpoint_model[k]
#
# # for key in ['head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias']:
# #     if key in checkpoint_model:
# #         print(f"Removing key {key} from pretrained checkpoint")
# #         del checkpoint_model[key]
#
# interpolate_pos_embed(model, checkpoint_model)
#
# # load pre-trained model
# msg = model.load_state_dict(checkpoint_model, strict=False)
#
# assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
#
# trunc_normal_(model.head.weight, std=2e-5)

model.load_state_dict(checkpoint_model)

print("Model = %s" % str(model))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

dataset_test = build_chr_dataset(is_train='test', args=args)

data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

# use my founction evaluate
# test_stats,auc_roc = evaluate(data_loader_test, model, device, args.task, epoch=0, mode='test',num_class=args.nb_classes)

# or write your own evaluation loop

criterion = torch.nn.CrossEntropyLoss()

model.eval()

for i, (inputs, targets) in enumerate(data_loader_test):
    inputs, targets = inputs.to(device), targets.to(device)
    true_label=F.one_hot(targets.to(torch.int64), num_classes=args.nb_classes)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    prediction_softmax = torch.nn.Softmax(dim=1)(outputs)
    _,prediction_decode = torch.max(prediction_softmax, 1)
    _,true_label_decode = torch.max(true_label, 1)

    pass
    # for visualization
