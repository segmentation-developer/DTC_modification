import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet_sdf import VNet
#from networks.FC_Densenet import FCDenseNet57
#from networks.discriminator import FC3DDiscriminator

from dataloaders import utils
from utils import ramps, losses, metrics
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.util import compute_sdf, compute_sdf_inside
import wandb

wandb.init(project="SSL_LA heart2", config ={
        #"batch size" :16,
        #"max_iterations":5000,
        #"encoder": 'resnet-34',
        #"weights": 'imagenet',
        "mdoel" :"DTC",
        #"aspp":"2,3,4,5",
        #"decoder_use_batchnorm" :" yes",
       #"decoder_attention" : " X ",
        #"padding" : "constant",
        "server" : 57,
        "image size": "112,112,80"
    })

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/LA_dataset/2018LA_Seg_TrainingSet', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/SSL_DTC_SDMinside2', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='random seed')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float,  default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
#snapshot_path = "../model/" + args.exp + \
#    "_{}labels_beta_{}/".format(
#        args.labelnum, args.beta)
snapshot_path = "../model/" + args.exp

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        model = VNet(n_channels=1, n_classes=num_classes-1,
                   normalization='batchnorm', has_dropout=True, has_residual=True).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum    # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 80))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs_tanh, outputs = model(volume_batch)             #outputs_tanh :gpu
            outputs_soft = torch.sigmoid(outputs)


            '''
            plt.figure("image", (18, 18))
            plt.subplot(2, 2, 1)
            plt.title("bs=1")
            plt.imshow(outputs_tanh.squeeze()[0][:, :, 60:61].detach().cpu(), cmap='gray')
            #plt.imshow(sq_tanh[0][ :, :, 40:41].detach().cpu(), cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("bs=2")
            plt.imshow(outputs_tanh.squeeze()[1][:, :, 60:61].detach().cpu(), cmap='gray')
            plt.show()'''


            outputs_tanh_inside  = outputs_tanh.double() #outputs_tanh_inside :cpu
            #outputs_tanh_inside =outputs_tanh_inside.cpu().numpy
            outputs_tanh_inside = torch.where(outputs_tanh_inside < 0., 0., outputs_tanh_inside)#outputs_tanh_inside :gpu
            #outputs_tanh_inside = torch.from_numpy(outputs_tanh_inside).float().cuda()

            #for l, slice in enumerate(slices):
            #    outputs_tanh_inside[slice] = 0
            '''
            plt.figure("tanh_inside", (27, 18))
            plt.subplot(2, 2, 1)
            plt.title("tanh_bs=1")
            plt.imshow(outputs_tanh_inside.squeeze()[0][:, :, 60:61].detach().cpu(), cmap='gray')
            # plt.imshow(sq_tanh[0][ :, :, 40:41].detach().cpu(), cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title("tanh_bs=2")
            plt.imshow(outputs_tanh_inside.squeeze()[1][:, :, 60:61].detach().cpu(), cmap='gray')
            plt.show()'''


            # calculate the loss
            with torch.no_grad():
                gt_dis = compute_sdf(label_batch[:].cpu().numpy(), outputs[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()


                '''
                plt.figure("gt", (18, 18))
                plt.subplot(2, 2, 1)
                plt.title("tanh_bs=1")
                plt.imshow(gt_dis.squeeze()[0][:, :, 60:61].detach().cpu(), cmap='gray')
                # plt.imshow(sq_tanh[0][ :, :, 40:41].detach().cpu(), cmap='gray')
                plt.subplot(2, 2, 2)
                plt.title("tanh_bs=2")
                plt.imshow(gt_dis.squeeze()[1][:, :, 60:61].detach().cpu(), cmap='gray')
                plt.subplot(2, 2, 3)
                plt.show()'''

                gt_dis_inside = compute_sdf_inside(label_batch[:].cpu().numpy(), outputs[:labeled_bs, 0, ...].shape)
                #gt_dis_inside = torch.from_numpy(gt_dis).float().cuda()

                #gt_dis_inside = gt_dis.double()               #gt_dis:gpu ,gt_dis_inside : gpu
                #gt_dis_inside = torch.where(gt_dis_inside > 0., 0., gt_dis_inside)
                #gt_dis_inside = torch.from_numpy(gt_dis_inside).float().cuda()
                #gt_dis_cpu = gt_dis.cpu().numpy()
                #gt_dis_double_cpu = gt_dis.double().cpu().numpy()
                #gt_dis_inside_cpu = gt_dis_inside.cpu().numpy()
                '''
                plt.figure("tanh_inside", (18, 18))
                plt.subplot(2, 2, 1)
                plt.title("tanh_bs=1")
                plt.imshow(gt_dis_inside[0][:, :, 60:61].detach().cpu(), cmap='gray')
                plt.subplot(2, 2, 2)
                plt.title("tanh_bs=2")
                plt.imshow(gt_dis_inside[1][:, :, 60:61].detach().cpu(), cmap='gray')
                plt.subplot(2, 2, 3)
                plt.show()'''


            loss_sdf = mse_loss(outputs_tanh_inside[:labeled_bs,0, ...], gt_dis_inside)
            loss_seg_ce = ce_loss(
                outputs[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice = losses.dice_loss(
                outputs_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)
            dis_to_mask = torch.sigmoid(-1500*outputs_tanh)

            consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
            supervised_loss = loss_seg_dice + loss_seg_ce + args.beta * loss_sdf
            consistency_weight = get_current_consistency_weight(iter_num//150)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dc = metrics.dice(torch.argmax(
                outputs_soft[:labeled_bs], dim=1), label_batch[:labeled_bs])

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg_ce, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('loss/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss',
                              consistency_loss, iter_num)


            wandb.log({
                "iter": iter_num,
                "total_loss": loss.item(),  # total loss
                "SL_seg_Loss": loss_seg_dice.item() + loss_seg_ce.item() ,  # segmentation loss :dice  :super
                "SL_SDM_loss": loss_sdf.item(),
                "consistency_weight": consistency_weight,
                "consistency_loss": consistency_loss.item(),  # consis loss  : unsuper +super
                # "random_drop_loss": random_drop_loss.item(),
                # "feature_drop_loss": feature_drop_loss.item(),
                # "noise_loss": noise_loss.item(),
                # "sl_auxDec_concat_loss" : sl_auxDec_loss.item(),
                # "unsl_auxDec_concat_loss": unsl_auxDec_loss.item(),
                # "dice_metrics": dc.item()

            })

            '''
            logging.info(
                'iteration %d : loss : %f, loss_consis: %f, loss_haus: %f, loss_seg: %f, loss_dice: %f' %
                (iter_num, loss.item(), consistency_loss.item(), loss_sdf.item(),
                 loss_seg_ce.item(), loss_seg_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = dis_to_mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Dis2Mask', grid_image, iter_num)

                image = outputs_tanh[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/DistMap', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

                image = gt_dis[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_DistMap',
                                 grid_image, iter_num)
                '''
            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
