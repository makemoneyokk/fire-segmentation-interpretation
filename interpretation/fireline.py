import cv2
import os
import numpy as np

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def save_contours(contours, folder_path, filename):
    output_folder = folder_path + 'contour/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    contour_img = np.zeros_like(img)


    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
    # contour_img = cv2.dilate(contour_img, None, iterations=2)

    save_path = os.path.join(output_folder, filename)

    cv2.imwrite(save_path, contour_img)
    print(f"Saved contours: {save_path}")


input_folder_path = "D:/overlay"
for filename in os.listdir(input_folder_path):
    img_path = os.path.join(input_folder_path, filename)
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours = find_contours(binary)
        save_contours(contours, input_folder_path, filename)
