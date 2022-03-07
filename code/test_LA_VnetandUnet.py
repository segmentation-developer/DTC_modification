import os
import argparse
import torch
#from networks.vnet_sdf import VNet
from test_util_noise import test_all_case
#from networks.vnet_sdf import VNet
#from networks.attentionNet import AttU_Net
#from networks.unet3D import unet_3D
#from networks.attention_Unet3D import Attention_UNet
from networks.unet3D import unet_3D
from networks.vnet_addNoise import VNet


print(torch.cuda.is_available())
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/LA_dataset/2018LA_Seg_TrainingSet', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='LA/SSL_DTC_VNetandUnet_addNoise', help='model_name')
parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1,
                    help='apply NMS post-procssing?')


FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
#snapshot_path = "../model/{}".format(FLAGS.model)
snapshot_path = "/home/sohui/code/DTC/model/{}".format(FLAGS.model)

num_classes = 2

test_save_path = os.path.join(snapshot_path, "test/")
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
              image_list]


def test_calculate_metric():
    net1 = VNet(n_channels=1, n_classes=num_classes - 1,
                normalization='batchnorm', has_dropout=True).cuda()
    net2 = unet_3D(in_channels=1, n_classes=1).cuda()
    # net = Attention_UNet(n_classes=num_classes-1,in_channels=1).cuda()

    # net = nn.DataParallel(net).to(device)
    # save_mode_path = os.path.join(
    #    snapshot_path, 'best_model.pth')
    save_mode_path1 = os.path.join(snapshot_path, 'model1_iter_6000.pth')
    save_mode_path2 = os.path.join(snapshot_path, 'model2_iter_6000.pth')
    net1.load_state_dict(torch.load(save_mode_path1))
    net2.load_state_dict(torch.load(save_mode_path2))

    print("init weight from {}".format(save_mode_path1))
    net1.eval()

    avg_metric1 = test_all_case(net1, image_list, num_classes=num_classes - 1,
                                patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                save_result=True, test_save_path=test_save_path,
                                metric_detail=FLAGS.detail, nms=FLAGS.nms)

    net1.train()

    print("init weight from {}".format(save_mode_path2))
    net2.eval()
    avg_metric2 = test_all_case(net2, image_list, num_classes=num_classes - 1,
                                patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                save_result=True, test_save_path=test_save_path,
                                metric_detail=FLAGS.detail, nms=FLAGS.nms)

    net2.train()

    if (avg_metric1[0] >= avg_metric2[0]):
        return avg_metric1
    else:
        return avg_metric2


if __name__ == '__main__':
    metric = test_calculate_metric()  # 6000
    print(metric)
