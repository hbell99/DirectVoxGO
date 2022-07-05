import cv2, imageio
import numpy as np

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


img_path = 'logs/tri_dvgo_multiscene_v1/nerf_synthetic/1resconv_liif_d_o_lego/render_test_fine_last_testdown_4/lego/060.png'
# img_path = 'logs/tri_dvgo_multiscene_v1/nerf_synthetic/1resconv_newrgb_d_o_lego/render_test_fine_last_testdown_4/lego/000.png'
img_path = 'logs/nerf_synthetic/dvgo_down4/render_test_fine_last/060.png'
# img_path = 'data/nerf_synthetic/lego/test/r_60.png'

def read_image(path):
    image = imageio.imread(path)
    image = (np.array(image) / 255.).astype(np.float32)
    if image.shape[-1] == 4:
        image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])
    return to8b(image)

# img = cv2.imread(img_path)
img = read_image(img_path)

x0 = 300
y0 = 300
x1 = 500
y1 = 500
crop_img = img[y0:y1,x0:x1]      #x0,y0为裁剪区域左上坐标；x1,y1为裁剪区域右下坐标
imageio.imwrite('060.png',crop_img)
# cv2.imwrite('050_gt.png',crop_img)