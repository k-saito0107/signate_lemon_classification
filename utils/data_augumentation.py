import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from torchvision.transforms import Compose, Lambda, ToTensor, Normalize


class Compose():
    def __init__(self, transform):
        self.transforms = transform
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        
        return img


class RandomMirror():
    def __call__(self, img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
        return img



class Enhance():
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, img):
        factor = np.random.uniform(self.factor[0], self.factor[1])
        img = ImageEnhance.Color(img)
        img = img.enhance(factor)
        return img

class Scale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):

        width = img.size[0]  # img.size=[幅][高さ]
        height = img.size[1]  # img.size=[幅][高さ]

        # 拡大倍率をランダムに設定
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)  # img.size=[幅][高さ]
        scaled_h = int(height * scale)  # img.size=[幅][高さ]
        #a = np.random.randint(5)
        #if a == 1 or a== 2 or a== 3 or a == 4:

        # 画像のリサイズ
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)


        # 画像を元の大きさに
        # 切り出し位置を求める
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
        return img

class RandomRotation():
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):

        # 回転角度を決める
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # 回転
        img = img.rotate(rotate_angle, Image.BILINEAR)

        return img


class Resize():
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def __call__(self, img):
        img = img.resize((self.width, self.height))
        #anno_img = anno_img.resize((self.width, self.height))

        return img


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img):

        # PIL画像をTensorに。大きさは最大1に規格化される
        img = transforms.functional.to_tensor(img)

        # 色情報の標準化
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        return img
