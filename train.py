import argparse
import numpy as np
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from model.lgss import LGSS
from metrics import calc_ap, calc_miou
from data.moviescenes import MovieScenes
from sklearn.metrics import average_precision_score

def train(args, model, data_args):
    """
    Trains the model.

    Args:
        args (argparse.Namespace): Parsed commandline arguments.
        model (nn.Module): Instantiated model.
        data_args (dict): Arguments for instantiating the dataloader.

    """
    # Instantiate training dataloader
    train_loader = torch.utils.data.DataLoader(MovieScenes('./data/train.txt', 'train', **data_args))

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15])
    criterion = nn.CrossEntropyLoss(torch.Tensor([0.5, 5]).cuda())

    running_loss = 0.0
    for epoch in range(args.epochs):
        model.train()

        for batch_idx, (data, labels) in enumerate(train_loader):
            data_place, data_cast, data_act, data_aud = data
            data_place = data_place.cuda()
            data_cast  = data_cast.cuda()
            data_act   = data_act.cuda()
            data_aud   = data_aud.cuda()
            labels = labels.view(-1).cuda() 

            optimizer.zero_grad()
            output = model(data_place, data_cast, data_act, data_aud)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0 and batch_idx != 0:
                print('epoch %d, mini-batch %d: running loss: %.5f' % (epoch + 1, batch_idx, running_loss / 2000))
                running_loss = 0.0
        scheduler.step()

    output_path = os.path.join(args.output, 'sceneseg.pth')
    torch.save(model.state_dict(), output_path)
    print('Finished training. Model is saved to {}.'.format(output_path))


def test(args, model, data_args):
    """
    Performs testing.

    Args:
    args (argparse.Namespace): Parsed commandline arguments.
        model (nn.Module): Instantiated model.
        data_args (dict): Arguments for instantiating the dataloader.

    """
    # Instantiate test dataloader
    test_loader = torch.utils.data.DataLoader(MovieScenes('./data/test.txt', 'test', **data_args))

    model.load_state_dict(torch.load(args.weights))
    model.eval()

    gt_dict = dict()
    pr_dict = dict()
    shot_end_frame_dict = dict()

    print('Evaluating the model...')
    with torch.no_grad():
        for (data, labels, imdb_id, shot_end_frame) in test_loader:
            data_place, data_cast, data_act, data_aud = data
            data_place = data_place.cuda()
            data_cast  = data_cast.cuda()
            data_act   = data_act.cuda()
            data_aud   = data_aud.cuda()
            labels = labels.view(-1).cuda() 

            output = model(data_place, data_cast, data_act, data_aud)
            output = output.view(-1, 2)
            output = F.softmax(output, dim=1)
            prob = output[:, 1]

            gt = labels.cpu().detach().numpy()
            prediction = prob.cpu().detach().numpy()
            imdb_id = imdb_id[0]

            if any(gt):
                if imdb_id not in gt_dict:
                    gt_dict[imdb_id] = gt
                    pr_dict[imdb_id] = prediction
                else:
                    gt_dict[imdb_id] = np.concatenate((gt_dict[imdb_id], gt))
                    pr_dict[imdb_id] = np.concatenate((pr_dict[imdb_id], prediction))
                shot_end_frame_dict[imdb_id] = shot_end_frame[0]
            
    # Calculate mAP and Mean Miou
    AP, mAP, AP_dict = calc_ap(gt_dict, pr_dict)
    mean_miou, miou_dict = calc_miou(gt_dict, pr_dict, shot_end_frame_dict)
    print('mAP: %.5f' % (mAP))
    print('Mean Miou: %.3f' % (mean_miou))
    print('Finished testing.')


parser = argparse.ArgumentParser(description='Handle model and dataset args.')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Train or test.')
parser.add_argument('--batch-size', type=int, default=2, help='Number of samples in each batch')
parser.add_argument('--epochs', type=int, default=90, help='Number of training epochs.')
parser.add_argument('--seq-len', type=int, default=4, help='Sequence length.')
parser.add_argument('--shot-num', type=int, default=2, help='Number of shots.')
parser.add_argument('--output', type=str, default='weights', help='Output directory.')
parser.add_argument('--weights', type=str, default='weights/sceneseg.pth', help='Location of weightsfile.')


if __name__ == '__main__':
    args = parser.parse_args()

    model_args = {
        'seq_len': args.seq_len,
        'shot_num': args.shot_num, 
        'mode': ['place', 'cast', 'act', 'aud'],
        'sim_channel': 512, 
        'place_feat_dim': 2048,
        'cast_feat_dim': 512,
        'act_feat_dim': 512,
        'aud_feat_dim': 512,
        'aud': {'cos_channel':512},
        'bidirectional': True,
        'lstm_hidden_size': 512,
        'ratio': [0.5, 0.2, 0.2, 0.1]
    }

    # Instantiate model
    model = LGSS(model_args)
    model = nn.DataParallel(model)

    data_args = {
        'seq_len': args.seq_len,
        'shot_num': args.shot_num,
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 2
    }
    
    if args.mode == 'train':
        train(args, model, data_args)
    else:
        test(args, model, data_args)