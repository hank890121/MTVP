## 处理pred结果的.json文件,画图
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))

def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False):
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        color_area[result[0] == 1] = [128, 64, 128]
        color_area[result[1] ==1] = [128, 64, 128]
        # color_area[result[0] == 1] = [0, 255, 0]
        # color_area[result[1] ==1] = [0, 0, 255]
        color_seg = color_area

    color_mask = np.mean(color_seg, 2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 只保留高飽和高亮度的區域（代表鮮豔色線條）
    preserve_mask = (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 100)  # S > 100 且 V > 100

    # Step 2: 合成遮罩邏輯
    blend_mask = (color_mask != 0) & (~preserve_mask)

    # Step 3: 疊加圖像
    blended_img = img.copy()
    blended_img[blend_mask] = (
        img[blend_mask] * 0.6 + color_seg[blend_mask] * 0.4
    ).astype(np.uint8)
    img = blended_img
    # red_condition = np.all(img == [0, 255, 0], axis=-1)
    # img[~red_condition & (color_mask != 0)] = img[~red_condition & (color_mask != 0)] * 0.8 + color_seg[~red_condition & (color_mask != 0)] * 0.2
    # img[color_mask != 0] = img[color_mask != 0 ] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img.astype(np.uint8)
    # img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_segresult.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_segresult.png".format(epoch,index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_seg_gt.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_seg_gt.png".format(epoch,index), img)  
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    pass