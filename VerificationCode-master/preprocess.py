# -*- coding: utf-8 -*-
# @Time  : 2021/3/20 11:41
# @Author : zhoujiangtao
# @Desc : ==============================================
# Life is Short I Use Python!!!                      
# If this runs wrong,don't ask me,I don't know why;  
# If this runs right,thank god,and I don't know why. 
# Maybe the answer,my friend,is blowing in the wind. 
# ======================================================
# 对原始文件夹中的图片进行大小调整和重命名，分割图像放入各自的文件夹
from skimage import io
import torch
import matplotlib.pyplot as plt
from PIL import Image

def show_img(img_t,title = "image"):
    plt.imshow(img_t, cmap="gray")
    plt.title(title)
    plt.show()


def get_gray_img(img_f):
    img_t = torch.from_numpy(io.imread(img_f, as_gray=True))
    return img_t


def show_imgtensor_message(img_t):
    print(img_t)
    print("shape:{}".format(img_t.shape))
    img_t_flat = img_t.view(1, -1)
    print("max:{}".format(img_t_flat.max(dim=1)))
    print("min:{}".format(img_t_flat.min(dim=1)))


def binarization(img_t):
    z = torch.zeros(40, 80).double()
    o = torch.ones(40, 80).double()
    img_t_b = torch.where(img_t < img_t.mean().item(), z, o)
    return img_t_b


def median_filter(t, kennel=3):
    w = t.shape[1]
    h = t.shape[0]
    for i in range(h - kennel):
        for j in range(w - kennel):
            t[i, j] = t[i:i + kennel, j:j + kennel].median().float().item()
    return t

# margin分别为图片的[上，下，左，右]
def spilt(t, label, margin, slice=4):
    ls_image = []
    ls_lab = []
    for i in range(slice):
        s_p = t[0 + margin[0]:40 - margin[1], 20 * i + margin[2]:20 * (i + 1) - margin[3]]
        ls_image.append(s_p)
        ls_lab.append(label[i])
    return ls_image, ls_lab

def show_images(images,labels):
    plt.figure()
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.suptitle(labels)
        plt.imshow(images[i])
    plt.show()

import os

def preprocess_images(image_path = "./data/train/",save2 = "./data/image_train/"):
    fs = os.listdir(image_path)
    for f in fs:
        img_t = get_gray_img("{}{}".format(image_path,f))
        img_t_b = binarization(img_t)
        img_t_m = median_filter(img_t_b)
        image_name = f.replace("png","")
        images, labels = spilt(img_t_m, image_name, margin=[5, 5, 1, 1])
        for image,label in zip(images, labels):
            lab_dir = "{}{}".format(save2,label)
            if(not os.path.exists(lab_dir)):
                os.mkdir(lab_dir)
            # 保存图片需要转化为像素值，灰度值*255即可
            image = (image * 255).type(dtype=torch.uint8)
            # 将2a45中的a图片保存成形如2a45_2.png的形式，方便以后追述
            image_f = "{}/{}_{}.png".format(lab_dir,f.replace("png",""),label)
            io.imsave(image_f,image)


def resize_images_in_folder(folder_path, output_folder_path, size=(80, 40)):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 查找文件名中_的位置
            underscore_index = filename.find('_')

            # 如果找到了_
            if underscore_index != -1:
                # 构建新的文件名（去掉_之后的部分）
                new_filename = filename[:underscore_index]+'.png'

                # 构造原始文件和新文件的完整路径
                old_file_path = os.path.join(root, filename)
                new_file_path = os.path.join(root, new_filename)

                # 检查新文件名是否已经存在
                if not os.path.exists(new_file_path):
                    try:
                        # 重命名文件
                        os.rename(old_file_path, new_file_path)
                        #print(f"Renamed '{old_file_path}' to '{new_file_path}'")
                    except Exception as e:
                        pass
                        #print(f"Error renaming '{old_file_path}': {e}")
                else:
                    pass
                    #print(f"New file name '{new_filename}' already exists in '{root}'.")
            else:
                pass
                # 如果没有找到_，则不处理
                #print(f"No underscore found in filename '{filename}'")
                # 确保输出文件夹存在
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

        # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 忽略非图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 读取原始图片
            img = Image.open(os.path.join(folder_path, filename))
            # 调整图片大小
            resized_img = img.resize(size,
                                     Image.LANCZOS)  # 注意：应该是 Image.ANTIALIAS 或 Image.ANTIALIASED，取决于你的Pillow版本
            # 保存调整大小后的图片到输出文件夹
            resized_img.save(os.path.join(output_folder_path, filename), img.format)

        # 使用示例




if (__name__ == "__main__"):
    folder_path = './data/train/'  # 替换为你的图片文件夹路径
    output_folder_path = './data/train/'  # 替换为你想要保存调整大小后图片的文件夹路径
    resize_images_in_folder(folder_path, output_folder_path)

    #preprocess_images()
    preprocess_images(image_path="./data/train/", save2="./data/image_train/")
    # preprocess_images(image_path = "./data/test/",save2="./data/image_test/")
    # img = "./data/new/3vjj.png"
    # img_t = get_gray_img(img)
    # show_img(img_t)
    # show_imgtensor_message(img_t)
    # img_t_b = binarization(img_t)
    # show_img(img_t_b)
    # img_t_m = median_filter(img_t_b)
    # show_img(img_t_b)
    # images, labels = spilt(img_t_m, "2a45", margin=[5, 5, 1, 1])
    # show_images(images,labels)