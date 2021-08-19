import argparse
import logging

import torch.utils.data

from models.common import post_process_output
from utils.dataset_processing import evaluation
from utils.data.oran_data import OranDataset
from models.RgbdGGCNN import *

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.1,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    # Load Network
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    test_dataset = OranDataset(args.dataset_path, start=0, end=1.0)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )
    logging.info('Done')

    results = {'correct': 0, 'failed': 0}
    with torch.no_grad():
        for x, y, z,idx,gt_A in test_data:
            logging.info('Processing {}/{}'.format(idx + 1, len(test_data)))
            xc = x.to(device)  # rgb
            yc = y.to(device)  # d
            zc = [zz.to(device) for zz in z]  # gt
            lossd = net.compute_loss(xc, yc, zc)

            q_img, ang_img,sin_img,cos_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'])
            if args.vis:
                evaluation.plot_output(xc.cpu().numpy().squeeze().transpose(1,2,0),
                                       yc.cpu().numpy().squeeze().transpose(1,2,0), q_img,
                                       ang_img,sin_img,cos_img)

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                  results['correct'] + results['failed'],
                                                  results['correct'] / (results['correct'] + results['failed'])))

