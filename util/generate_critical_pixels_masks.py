import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
from PIL import Image
import cv2 as cv
from util.dncnn.network_dncnn import DnCNN as net
from util.dncnn import utils_image as util
import torch

from torchvision import transforms
import numpy as np

img_size = 1024
n_classes = 19
device = 'cpu'
# file_path = 'E:\\MachineLearning\\gan-compression'
file_path = '/root/ML'


def get_edge_mask_by_canny(img):
    '''
    通过canny获得图像边缘
    :param img:彩色图片
    :return:边缘标记
    '''
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(img, 30, 100)
    return edges


def get_mask_by_CAR(model, img_l, img_h, device):
    img_l = util.uint2single(img_l)
    img_l = util.single2tensor4(img_l)
    img_e = model(img_l)
    img_e = util.tensor2uint(img_e)
    img_h = img_h.squeeze()
    err = cv.absdiff(img_e, img_h)
    err = cv.cvtColor(err, cv.COLOR_BGR2GRAY)
    err = cv.threshold(err, 5, 255, cv.THRESH_BINARY)
    return np.asarray(err)


def Extract_Face_Mask(pil_image, parsing_net, to_tensor, device):
    '''
    Usage:
        Extract the face foreground from an pil image

    Args:
        pil_image:   (PIL.Image) a single image
        parsing_net: (nn.Module) the network to parse the face images
        to_tensor:   (torchvision.transforms) the image transformation function
        device:      (str) device to place the networks
    '''

    with torch.no_grad():
        # image = pil_image.resize((512, 512), Image.BILINEAR)
        # img = to_tensor(image)
        img = to_tensor(pil_image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = parsing_net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    return parsing


def get_roi_mask(img, parsing_net, to_tensor, device):
    single_img = to_tensor(img)
    parsing = Extract_Face_Mask(img, parsing_net, to_tensor, device)
    mask = (parsing > 0) * (parsing != 16)
    resized_mask = np.array(Image.fromarray(np.uint8(mask)).resize((img_size, img_size)),dtype=object)
    return resized_mask


def get_critical_mask(img_l, img_h, car_model, parsing_net, device):
    for k, v in car_model.named_parameters():
        v.requires_grad = False
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    roi_mask = get_roi_mask(img_h, parsing_net, to_tensor, device)
    car_mask = get_mask_by_CAR(car_model, img_l, img_h, device)
    edge_mask = get_edge_mask_by_canny(img_h)
    mask = np.zeros([img_size, img_size], dtype=int)
    for i in range(img_size):
        for j in range(img_size):
            if roi_mask[i][j] == True and car_mask[1][i][j] == 255 and edge_mask[i][j] == 255:
                mask[i, j] = 255
    return mask


if __name__ == '__main__':
    n_classes = 19
    from util.face_parsing.BiSeNet import BiSeNet

    n_channels = 3
    nb = 20
    device = 'cpu'
    # 加载dncnn网络
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    # model_path = image_path = file_path + '''\\util\\model_zoo\\dncnn_color_blind.pth'''
    model_path = image_path = file_path + '''/util/model_zoo/dncnn_color_blind.pth'''
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    # 加载parsing_net网络
    # PRETRAINED_FILE = (file_path + '''\\util\\face_parsing\\pretrained_model\\79999_iter.pth''')
    PRETRAINED_FILE = (file_path + '''/util/face_parsing/pretrained_model/79999_iter.pth''')
    parsing_net = BiSeNet(n_classes=n_classes).to(device)
    pretrained_weight = torch.load(PRETRAINED_FILE, map_location=device)
    parsing_net.load_state_dict(pretrained_weight)
    parsing_net.eval()
    for i in range(200,29999):
        # L_path = os.path.join(file_path + "\\util\\test_sets\\low", str(i) + '_low.jpg')
        L_path = os.path.join(file_path + "/util/test_sets/low", str(i) + '_low.jpg')
        # H_path = os.path.join(file_path + "\\data\\CelebAMask-HQ\\CelebA-HQ-img\\", str(i) + '.jpg')
        H_path = os.path.join(file_path + "/data/CelebAMask-HQ/CelebA-HQ-img/", str(i) + '.jpg')
        # s_path = os.path.join(file_path + "\\util\\test_sets\\mask", str(i) + '_mask.jpg')
        s_path = os.path.join(file_path + "/util/test_sets/mask", str(i) + '_mask.jpg')
        # print(L_path)
        # 生成压缩图像并读取
        img = Image.open(H_path)
        if img.mode=='I':
            img = img.convert("RGB")
        img.save(L_path, quality=10)
        img_h = util.imread_uint(H_path, n_channels=n_channels)
        img_l = util.imread_uint(L_path, n_channels=n_channels)
        mask = get_critical_mask(img_l, img_h, model, parsing_net, device)
        mask_img = Image.fromarray(np.uint8(mask))
        if mask_img.mode=='I':
            mask_img = mask_img.convert("RGB")
        mask_img.save(s_path)

