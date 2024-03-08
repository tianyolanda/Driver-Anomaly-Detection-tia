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

import numpy as np
import sys
from threading import Lock, Thread
from time import sleep

import cv2

# ZED imports
import pyzed.sl as sl

def load_image_into_numpy_array(image):
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_depth_into_numpy_array(depth):
    ar = depth.get_data()
    ar = ar[:, :, 0:4]
    (im_height, im_width, channels) = depth.get_data().shape
    return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)

lock = Lock()
width = 704
height = 416
confidence = 0.35

image_np_global = np.zeros([width, height, 3], dtype=np.uint8)
depth_np_global = np.zeros([width, height, 4], dtype=np.float)

exit_signal = False
new_data = False


# ZED image capture thread function
def capture_thread_func(svo_filepath=None):
    global image_np_global, depth_np_global, exit_signal, new_data

    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    input_type = sl.InputType()
    if svo_filepath is not None:
        input_type.set_from_svo_file(svo_filepath)

    init_params = sl.InitParameters(input_t=input_type)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.coordinate_units = sl.UNIT.METER
    init_params.svo_real_time_mode = False

    # Open the camera
    err = zed.open(init_params)
    print(err)
    while err != sl.ERROR_CODE.SUCCESS:
        err = zed.open(init_params)
        print(err)
        sleep(1)

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    image_size = sl.Resolution(width, height)

    while not exit_signal:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT, resolution=image_size)
            zed.retrieve_measure(depth_mat, sl.MEASURE.XYZRGBA, resolution=image_size)
            lock.acquire()
            image_np_global = load_image_into_numpy_array(image_mat)
            depth_np_global = load_depth_into_numpy_array(depth_mat)
            new_data = True
            lock.release()

        sleep(0.01)

    zed.close()

def main(args):

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

    svo_filepath = None
    if len(args) > 1:
        svo_filepath = args[1]

    print("Starting the ZED")
    capture_thread = Thread(target=capture_thread_func, kwargs={'svo_filepath': svo_filepath})
    capture_thread.start()
    # Shared resources
    global image_np_global, depth_np_global, new_data, exit_signal
    depth_img_seq = np.zeros((1, 1, 16, 112, 112))

    while not exit_signal:
        depth_img_ori = depth_np_global[:, 144:560, 0]  #　原本是 [416,704,4] --> [416,416,4]
        depth_img_ori = cv2.resize(depth_img_ori, (112, 112))
        depth_img=np.expand_dims(depth_img_ori, axis=0)
        depth_img=np.expand_dims(depth_img, axis=0)
        depth_img=np.expand_dims(depth_img, axis=0) # (1, 1, 16, 112, 112)

        depth_img_seq_rmfirst = depth_img_seq[:,:,1:,:,:]# (1, 1, 15, 112, 112) 去掉序列中的第1帧
        depth_img_seq_new = np.concatenate((depth_img_seq_rmfirst,depth_img),axis=2) # (1, 1, 16, 112, 112) 把最新一帧加入进去
        depth_img_seq_new =  torch.from_numpy(depth_img_seq_new)
        depth_img_seq_new = depth_img_seq_new.type(torch.float)

        if use_cuda:
            depth_img_seq_new = depth_img_seq_new.cuda()

        time_2 = time.time()
        out_1 = model_front_d(depth_img_seq_new)[1].detach()
        # out_2 = model_front_ir(data2[0])[1].detach()
        # out_3 = model_top_d(data3[0])[1].detach()
        # out_4 = model_top_ir(data4[0])[1].detach()
        time_3 = time.time()

        sim_1 = torch.mm(out_1, normal_vec_front_d.t())

        sim = round(torch.mean(sim_1).cpu().item(), 2)
        if sim >= threshold:
            action = 'Normal'
        else:
            action = 'Distracted'

        img = cv2.resize(depth_img_ori,(224,171))
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
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit_signal = True

    exit_signal = True
    capture_thread.join()

if __name__ == '__main__':
    main(sys.argv)
