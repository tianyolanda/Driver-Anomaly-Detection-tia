import time

import numpy as np
import spatial_transforms
import os
from models import shufflenet, shufflenetv2, resnet, mobilenetv2se,mobilenetv2
import torch.nn as nn
import torch
import cv2
from dataset_test import DAD_Test
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

#==========================================================================================================
# This file is used after training and testing phase to demonstrate a running demo of the Driver Monitoring System
# with our contrastive approach.
#
# The demo is based on four well-trained 3D-ResNet-18 architectures, and four normal driving template vectors have been
# produced by these architectures at test time.
#
# When the code is running, a window will shows the current frame (8th frame of the test video clip) with predicted
# label; the path of the current frame, the similarity score, and predicted actions will be shown in terminal synchronized.
#
# If 'delay' is set to 0, the code runs one sample at a time. By pressing any key, the next sample will be processed
# If 'delay' is not 0, the code will process all samples from be beginning with a delay of 'delay' millisecond
#==========================================================================================================

#======================================Hyperparameters=====================================================
root_path = '/home/ubuntu/datasets/DADdataset/DAD/DAD/'  #root path of the dataset
show_which = 'front_depth'  # show which view or modalities: {'front_depth', 'front_IR', 'top_depth', 'top_IR'}
threshold = 0.34  # the threshold
delay = 1  # The sample will be processed and shown with a delay of 'delay' ms.
           # If delay = 0, The code runs one sample at a time, press any key to process the next sample

sample_size = 112
sample_duration = 16
val_batch_size = 1
n_threads = 0
use_cuda = True # False
shortcut_type = 'A'
feature_dim = 512
width_mult=1.0

print('========================================Loading Normal Vectors========================================')
normal_vec_front_d = np.load('./normvec/normal_vec_front_d.npy')
# normal_vec_front_ir = np.load('./normvec/normal_vec_front_ir.npy')
# normal_vec_top_d = np.load('./normvec/normal_vec_top_d.npy')
# normal_vec_top_ir = np.load('./normvec/normal_vec_top_ir.npy')

normal_vec_front_d = torch.from_numpy(normal_vec_front_d)
# normal_vec_front_ir = torch.from_numpy(normal_vec_front_ir)
# normal_vec_top_d = torch.from_numpy(normal_vec_top_d)
# normal_vec_top_ir = torch.from_numpy(normal_vec_top_ir)

if use_cuda:
    normal_vec_front_d = normal_vec_front_d.cuda()
    # normal_vec_front_ir = normal_vec_front_ir.cuda()
    # normal_vec_top_d = normal_vec_top_d.cuda()
    # normal_vec_top_ir = normal_vec_top_ir.cuda()

val_spatial_transform = spatial_transforms.Compose([
    spatial_transforms.Scale(sample_size),
    spatial_transforms.CenterCrop(sample_size),
    spatial_transforms.ToTensor(255),
    spatial_transforms.Normalize([0], [1]),
])

print("===========================================Loading Test Data==========================================")

test_data_front_d = DAD_Test(root_path=root_path,
    subset='validation',
    view='front_depth',
    sample_duration=sample_duration,
    type=None,
    spatial_transform=val_spatial_transform,
)
test_loader_front_d = torch.utils.data.DataLoader(
    test_data_front_d,
    batch_size = val_batch_size,
    shuffle = False,
    num_workers = n_threads,
    pin_memory = True,
)
num_val_data_front_d = test_data_front_d.__len__()
print('Front depth view is done')
print('normal_vec_front_d',normal_vec_front_d.size())
#
# test_data_front_ir = DAD_Test(root_path=root_path,
#     subset = 'validation',
#     view = 'front_IR',
#     sample_duration = sample_duration,
#     type = None,
#     spatial_transform = val_spatial_transform,
# )
# test_loader_front_ir = torch.utils.data.DataLoader(
#     test_data_front_ir,
#     batch_size = val_batch_size,
#     shuffle = False,
#     num_workers = n_threads,
#     pin_memory = True,
# )
# num_val_data_front_ir = test_data_front_ir.__len__()
# print('Front IR view is done')
#
# test_data_top_d = DAD_Test(root_path=root_path,
#     subset = 'validation',
#     view = 'top_depth',
#     sample_duration = sample_duration,
#     type = None,
#     spatial_transform = val_spatial_transform,
# )
# test_loader_top_d = torch.utils.data.DataLoader(
#     test_data_top_d,
#     batch_size = val_batch_size,
#     shuffle = False,
#     num_workers = n_threads,
#     pin_memory = True,
# )
# num_val_data_top_d = test_data_top_d.__len__()
# print('Top depth view is done')
# #
# test_data_top_ir = DAD_Test(root_path=root_path,
#     subset = 'validation',
#     view = 'top_IR',
#     sample_duration = sample_duration,
#     type = None,
#     spatial_transform = val_spatial_transform,)
# test_loader_top_ir = torch.utils.data.DataLoader(
#     test_data_top_ir,
#     batch_size=val_batch_size,
#     shuffle=False,
#     num_workers=n_threads,
#     pin_memory=True,
# )
# num_val_data_top_ir = test_data_top_ir.__len__()
# print('Top IR view is done')
# assert num_val_data_front_d == num_val_data_front_ir == num_val_data_top_d == num_val_data_top_ir

print('=============================================Loading Models===========================================')

# model_front_d = resnet.resnet18(
#     output_dim=feature_dim,
#     sample_size=sample_size,
#     sample_duration=sample_duration,
#     shortcut_type=shortcut_type,
# )
#
model_front_d = mobilenetv2se.get_model(
    sample_size=sample_size,
    width_mult=width_mult,
)

# model_front_d = mobilenetv2.get_model(
#     sample_size=sample_size,
#     width_mult=width_mult,
# )

print('1111')
# model_front_ir = resnet.resnet18(
#     output_dim=feature_dim,
#     sample_size=sample_size,
#     sample_duration=sample_duration,
#     shortcut_type=shortcut_type,
# )
# print('22222')
#
# model_top_d = resnet.resnet18(
#     output_dim=feature_dim,
#     sample_size=sample_size,
#     sample_duration=sample_duration,
#     shortcut_type=shortcut_type,
# )
#
# model_top_ir = resnet.resnet18(
#     output_dim=feature_dim,
#     sample_size=sample_size,
#     sample_duration=sample_duration,
#     shortcut_type=shortcut_type,
# )
# print('33333')

model_front_d = nn.DataParallel(model_front_d, device_ids=None)
# model_front_ir = nn.DataParallel(model_front_ir, device_ids=None)
# model_top_d = nn.DataParallel(model_top_d, device_ids=None)
# model_top_ir = nn.DataParallel(model_top_ir, device_ids=None)
# print('44444')

resume_path_front_d = './checkpoints/mobilenetv2se_reduction4/best_model_mobilenetv2se_front_depth.pth'
# resume_path_front_d = './checkpoints/mobilenetv2_bs_small/best_model_mobilenetv2_front_depth.pth'
# resume_path_front_d = './checkpoints/resnet18_frontdepth/best_model_resnet_front_depth.pth'
# resume_path_front_d = 'checkpoints/mobilenetv2se_nofirst_reduction32/best_model_mobilenetv2se_front_depth.pth'
# resume_path_front_ir = './checkpoints/best_model_resnet_front_IR.pth'
# resume_path_top_d = './checkpoints/best_model_resnet_top_depth.pth'
# resume_path_top_ir = './checkpoints/best_model_resnet_top_IR.pth'

resume_checkpoint_front_d = torch.load(resume_path_front_d)
# resume_checkpoint_front_ir = torch.load(resume_path_front_ir)
# resume_checkpoint_top_d = torch.load(resume_path_top_d)
# resume_checkpoint_top_ir = torch.load(resume_path_top_ir)
print('55555')

model_front_d.load_state_dict(resume_checkpoint_front_d['state_dict'])
# model_front_ir.load_state_dict(resume_checkpoint_front_ir['state_dict'])
# model_top_d.load_state_dict(resume_checkpoint_top_d['state_dict'])
# model_top_ir.load_state_dict(resume_checkpoint_top_ir['state_dict'])
print('66666')

model_front_d.eval()
# model_front_ir.eval()
# model_top_d.eval()
# model_top_ir.eval()

frame_all=[]
sim_all=[]
# x_lim = [round(x * 0.2 - 1, 1) for x in range(11)]
# print(x_lim)
framenum_of_save_record = 2500

print('===========================================Calculating Scores=========================================')
for batch, data_ori in enumerate(zip(test_loader_front_d)):
    time_start = time.time()
    data1 = data_ori[0]
    if use_cuda:
        data1[0] = data1[0].cuda()
        data1[1] = data1[1].cuda()

    # assert torch.sum(data1[1])  == \
    #        data1[1].size(0)
    time_2 = time.time()
    out_1 = model_front_d(data1[0])[1].detach()
    # out_2 = model_front_ir(data2[0])[1].detach()
    # out_3 = model_top_d(data3[0])[1].detach()
    # out_4 = model_top_ir(data4[0])[1].detach()
    time_3 = time.time()

    sim_1 = torch.mm(out_1, normal_vec_front_d.t())

    # sim_2 = torch.mm(out_2, normal_vec_front_ir.t())
    # sim_3 = torch.mm(out_3, normal_vec_top_d.t())
    # sim_4 = torch.mm(out_4, normal_vec_top_ir.t())
    # sim = round(torch.mean(torch.stack((sim_1, sim_2, sim_3, sim_4), dim=0)).cpu().item(), 2)
    sim = round(torch.mean(sim_1).cpu().item(), 2)
    if sim >= threshold:
        action = 'Normal'
    else:
        action = 'Distracted'
    time_end = time.time()

    # record live demo running speed
    time_pre = round(time_2-time_start, 6)
    time_model = round(time_3-time_2, 6)
    time_sim = round(time_end - time_3, 6)
    time_all = round(time_end - time_start, 6)

    print('time_pre',time_pre)
    print('time_model',time_model)
    print('time_sim',time_sim)
    print('time_all',time_all)

    folder = int(batch // 60000) + 1
    subfolder = int((batch % 60000) // 10000) + 1

    index = (batch % 60000) % 10000
    img_path = os.path.join(root_path, 'val0'+str(folder)+'/rec'+str(subfolder)+'/'+show_which+'/img_'+str(index)+'.png')
    # img_path = '/home/ubuntu/datasets/DADdataset/DAD/Tester1/adjusting_radio/front_depth'+'/img_'+str(index)+'.png'

    print(f'Img: {img_path} | score: {sim} | Action: {action}')

    img = cv2.imread(img_path)
    print('img.shape',img.shape)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 'Score: ' + str(sim)
    img = cv2.putText(img, score, (80, 20), font, 0.8, (0, 0, 255), 1)
    if action == 'Normal':
        img = cv2.putText(img, action, (93, 35), font, 0.8, (0, 0, 255), 1)
    elif action == 'Distracted':
        img = cv2.putText(img, action, (83, 35), font, 0.8, (0, 0, 255), 1)
    cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
    cv2.imshow('Demo', img)
    cv2.waitKey(delay)

    # record frame-sim
    frame_all.append(index)
    sim_all.append(sim)
    rec_slice = int((batch % 60000) / framenum_of_save_record) # 0,1,2,3
    print('rec_slice, index,sim',rec_slice,index,sim)
    if index % framenum_of_save_record == 0:
        plt.figure(figsize=(16,9))
        print(frame_all, sim_all)
        plt.xlabel('Frame ID')
        plt.ylabel('Similarity Score')
        x_major_locator = MultipleLocator(framenum_of_save_record/10)
        # 把x轴的刻度间隔设置为1，并存在变量里
        y_major_locator = MultipleLocator(0.2)
        # 把y轴的刻度间隔设置为10，并存在变量里
        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        ax.spines['top'].set_visible(False)  # 设置去掉上边框
        ax.spines['right'].set_visible(False)  # 设置去掉右边框

        plt.xlim([(rec_slice-1)*framenum_of_save_record, rec_slice*framenum_of_save_record])
        plt.ylim([-1, 1])
        plt.plot(frame_all, sim_all,color='blue')

        # plt.show()
        framename = 'val0'+str(folder)+'_rec'+str(subfolder)+'_'+show_which+'_slice'+str(rec_slice)+'.png'
        plt.savefig('./result/frame_similarity/'+ framename)
        # exit()
        frame_all = []
        sim_all = []

cv2.destroyAllWindows()

