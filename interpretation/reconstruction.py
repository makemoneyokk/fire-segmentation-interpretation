import cv2
import os
import numpy as np
def overlay_images(folder1_path, folder2_path, output_folder, thickness=5):

    folder1_images = [os.path.join(folder1_path, file) for file in sorted(os.listdir(folder1_path)) if file.endswith('.png') or file.endswith('.jpg')]
    folder2_images = [os.path.join(folder2_path, file) for file in sorted(os.listdir(folder2_path)) if file.endswith('.png') or file.endswith('.jpg')]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(min(len(folder1_images), len(folder2_images))):
        image1 = cv2.imread(folder1_images[i])

        image2 = cv2.imread(folder2_images[i])

        overlay = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
        intersection = cv2.bitwise_and(image1, image2)
        intersection[intersection[:, :, 0] > 0] = [0, 0, 255]

        kernel = np.ones((thickness, thickness), np.uint8)
        intersection = cv2.dilate(intersection, kernel, iterations=1)

        result = cv2.addWeighted(overlay, 1, intersection, 1, 0)

        output_path = os.path.join(output_folder, f"overlay_{i}.png")
        cv2.imwrite(output_path, result)

    print("over")


folder1_path = "D:/the/folder/of/time accumulation "
folder2_path = "D:/the/folder/of/overlay"
output_folder = "D:/the/folder/of/reconstruction"

overlay_images(folder1_path, folder2_path, output_folder, thickness=10)






