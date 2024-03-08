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
        depth_img = depth_np_global[:, 144:560, 0] #　原本是 [416,704,4]
        depth_img = cv2.resize(depth_img, (112, 112))
        # depth_img = np.resize(1,1,1,112,112)
        depth_img=np.expand_dims(depth_img, axis=0)
        depth_img=np.expand_dims(depth_img, axis=0)
        depth_img=np.expand_dims(depth_img, axis=0)

        print('depth_img', depth_img.shape)
        # depth_img = cv2.resize(depth_np_global, (112, 112))

        depth_img_seq_rmfirst = depth_img_seq[:,:,1:,:,:]
        depth_img_seq_new = np.concatenate((depth_img_seq_rmfirst,depth_img),axis=2)
        print(depth_img_seq_new.shape)


        cv2.imshow('ZED image', depth_img[0,0,0,:,:])
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit_signal = True

    exit_signal = True
    capture_thread.join()


if __name__ == '__main__':
    main(sys.argv)
