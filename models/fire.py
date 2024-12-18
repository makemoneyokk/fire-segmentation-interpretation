import cv2
import os

def overlay_images(image_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of image files in the folder
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Initialize the overlay result as the first image
    overlay_result = first_image.copy()

    # Overlay each image and save the result
    for i in range(1, len(image_files)):
        image_path = os.path.join(image_folder, image_files[i])
        frame = cv2.imread(image_path)

        # Ensure the image was read successfully
        if frame is not None:
            # Overlay the current frame with the previous result
            overlay_result = cv2.addWeighted(overlay_result, 0.8, frame, 0.8, 0)

            # Save the overlay result
            output_path = os.path.join(output_folder, f'overlay_result_{i}.jpg')
            cv2.imwrite(output_path, overlay_result)

# Input image folder path and output folder path
image_folder_path = 'D:/data(teacher)/paper-code/YOLOv7-Pytorch-Segmentation/YOLOv7-Pytorch-Segmentation/runs/predict-seg/exp17'
output_folder_path = 'D:/data(teacher)/paper-code/YOLOv7-Pytorch-Segmentation/YOLOv7-Pytorch-Segmentation/runs/mask4'

# Overlay images and save the results
overlay_images(image_folder_path, output_folder_path)







