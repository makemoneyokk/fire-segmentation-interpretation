
import cv2
import numpy as np
import os


point1 = (281, 41)
point2 = (1181,157 )
point3 = (501, 1401)

# point1 = (0, 0)
# point2 = (573, 0 )
# point3 = (573, 600)
# point4 = (0, 600)

background_color = (0, 0, 0)  # 黑色
output_folder = "D:/Segmentation/runs/quadrilater_outdoor"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取文件夹中的图片并处理
input_folder = "D:/Segmentation/runs/outdoor_overlay1contour"
for filename in os.listdir(input_folder):
    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    mask = np.zeros_like(img)

    # triangle = np.array([point1, point2, point3, point4])
    triangle = np.array([point1, point2, point3])
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if cv2.pointPolygonTest(triangle, (x, y), False) >= 0:
                mask[y, x] = img[y, x]
            else:
                mask[y, x] = background_color

    # 保存处理后的图像至目标文件夹
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, mask)

    print(f"Saved image with triangle region in {output_path}")

print("Process completed.")




