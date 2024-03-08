import os
import cv2
# /home/ubuntu/Pictures/driver-anomal-det-result2/

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        #print(filename) #just for test
        #img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename)
        print(img.shape)
        img_cut = img[83:331,100:428,:]
        cv2.imshow('img_cut',img_cut)
        # cv2.waitKey(0)
        save_location = "/home/ubuntu/Pictures/driver-anomal-det-result2-cut/"
        cv2.imwrite(save_location+filename,img_cut)
        # array_of_img.append(img)
        #print(img)
        # print(array_of_img)

read_directory("/home/ubuntu/Pictures/driver-anomal-det-result2/")
