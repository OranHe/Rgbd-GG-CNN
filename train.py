import datetime
import os
import sys
import argparse
import logging

import cv2

import torch
import torch.utils.data
import torch.optim as optim

from torchsummary import summary
from skimage.filters import gaussian
import tensorboardX

from utils.dataset_processing import evaluation
from utils.data.oran_data import OranDataset
from models.RgbdGGCNN import *

logging.basicConfig(level=logging.INFO)
def post_process_output(q_img, cos_img, sin_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    return q_img, ang_img

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='RgbdGGCNN', help='Network Name in .models')

    # Dataset & Data & Training
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=0, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=821, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=973, help='Validation Batches')

    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='output/tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, batches_per_epoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, z,_,gt_A in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device) #rgb
                yc = y.to(device) #d
                zc = [zz.to(device) for zz in z] #gt
                lossd = net.compute_loss(xc, yc, zc)

                loss = lossd['loss']
                results['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()/ld

                q_out, ang_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'])

                s = evaluation.calculate_iou_match(q_out, ang_out,zc[0],gt_A)

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, z, _, _ in train_data:
            batch_idx += 1
            print("\r" + "batch_idx:", batch_idx, end="", flush=True)
            #print("batch_idx: ", batch_idx)
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)  # rgb
            yc = y.to(device)  # d
            zc = [zz.to(device) for zz in z]
            lossd = net.compute_loss(xc, yc, zc)

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()

    # Vis window
    if args.vis:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    train_dataset = OranDataset(args.dataset_path, start=0.0, end=args.split)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    # get some random training images
    # images,depthimages,_,_,_ = iter(train_data).next()

    val_dataset = OranDataset(args.dataset_path, start=args.split, end=1.0)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')

    net = RgbdGGCNN()
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    logging.info('Done')

    # Print model architecture.
    summary(net, [[3, 600, 600], [3, 600, 600]])
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, [[3, 600, 600], [3, 600, 600]])
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.val_batches)
        logging.info('Validating test loss: '+ str(test_results['loss']))
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()
