import cv2
import numpy as np
import os

def overlay_images(input_folder, output_folder):
    file_names = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    base_image = cv2.imread(os.path.join(input_folder, file_names[0]), cv2.IMREAD_GRAYSCALE)
    base_image = np.where(base_image > 128, 255, 0).astype(np.uint8)

    for i in range(1, len(file_names)):
        next_image = cv2.imread(os.path.join(input_folder, file_names[i]), cv2.IMREAD_GRAYSCALE)
        next_image = np.where(next_image > 128, 255, 0).astype(np.uint8)

        base_image = cv2.bitwise_or(base_image, next_image)

        output_file = os.path.join(output_folder, f'overlay_{i + 1}.png')
        cv2.imwrite(output_file, base_image)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    input_folder = "D:/the/folder/of/segmentation"
    output_folder = "D:/the/folder/of/output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    overlay_images(input_folder, output_folder)
