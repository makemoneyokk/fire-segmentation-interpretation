
import cv2
import os


def invert_binary_images(input_folder, output_folder):

    file_names = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in file_names:
        image_path = os.path.join(input_folder, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        inverted_image = cv2.bitwise_not(image)

        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, inverted_image)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    input_folder = "D:/contour"
    output_folder = "D:/black-w"

    invert_binary_images(input_folder, output_folder)
