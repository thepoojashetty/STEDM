import cv2
import numpy as np
#flowers
# images = [
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/flowers/segs/segmim_00017.jpg",
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/flowers/imgs/image_00063.jpg",
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/flowers/imgs/image_00070.jpg",
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/flowers/imgs/image_00083.jpg",
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/flowers/imgs/image_00095.jpg"
#     ]
# final_images = [
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/flowers/test_c.png",
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/flowers/augmented/0_img.png",
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/flowers/augmented/1_img.png",
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/flowers/augmented/2_img.png",
#         "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/flowers/augmented/3_img.png"
# ]

#cityscapes
images = [
    "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/cityscapes/test_5.png"
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/cityscapes/gtFine_trainvaltest/gtFine/val/frankfurt/frankfurt_000000_003025_gtFine_labelIds.png"
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/cityscapes/gtFine_trainvaltest/gtFine/train/bremen/bremen_000005_000019_gtFine_labelIds.png"
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/lindau/lindau_000002_000019_leftImg8bit.png",
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/munster/munster_000065_000019_leftImg8bit.png",
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/cityscapes/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/val/frankfurt/frankfurt_000000_000576_leftImg8bit_foggy_beta_0.01.png",
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/cityscapes/leftImg8bit_trainval_rain/leftImg8bit_rain/val/lindau/lindau_000003_000019_leftImg8bit_rain_alpha_0.01_beta_0.005_dropsize_0.01_pattern_1.png"
]

final_images = [
    "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/cityscapes/test_5.png"
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/cityscapes/test_c2.png"
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/cityscapes/augmented/0_img.png",
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/cityscapes/augmented/1_img.png",
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/cityscapes/augmented/2_img.png",
    # "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/cityscapes/augmented/3_img.png"
]

for i in range(len(images)):
    img = cv2.imread(images[i])
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    # img[img==26]=255
    # img[img!=255]=0
    cv2.imwrite(final_images[i], img)

#create an empty image
# img = np.zeros((512, 512, 3), np.uint8)
# cv2.imwrite("/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/cityscapes/test_3.png", img)

#create an image with all 255
# img = np.ones((512, 512, 3), np.uint8)*26
# cv2.imwrite("/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/cityscapes/test_4.png", img)