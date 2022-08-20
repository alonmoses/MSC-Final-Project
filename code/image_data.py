import scanpy as sc
import stlearn as st
import cv2
import os
from typing import Tuple
import pandas as pd


def tile_image(adata, out_path:str='./tiling', crop_size:int = 40):
    st.pp.tiling(adata, out_path, crop_size=crop_size)
    return adata

def generate_edge_detected_images(images_path:str, out_path:str, blur_kernel_size:Tuple=(3,3)):
    i = 0
    for image in os.scandir(images_path):
        img = cv2.imread(image.path)
        if i < 400:
            cv2.imwrite(f"{out_path}/tiling_classification_train/{i}.jpeg", img)
        i += 1
        # Convert to graycsale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, blur_kernel_size, 0)
        
        # Sobel Edge Detection
        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        # Display Sobel Edge Detection Images
        cv2.imwrite(f"{out_path}/edge_detect/x/{image.path.split('/')[-1]}", sobelx)
        cv2.imwrite(f"{out_path}/edge_detect/y/{image.path.split('/')[-1]}", sobely)
        cv2.imwrite(f"{out_path}/edge_detect/xy/{image.path.split('/')[-1]}", sobelxy)
        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
        # Display Canny Edge Detection Image
        cv2.imwrite(f"{out_path}/edge_detect/canny/{image.path.split('/')[-1]}", edges)


def tag_images_for_edges(adata, edge_detect_images:str, out_path:str):
    tagged_data = pd.DataFrame(columns=['image_path', 'has_edge'])
    for i, image in enumerate(os.scandir(edge_detect_images)):
        user_label = input(f"Enter label for image {i}: ")
        row = {'has_edge': user_label}
        tagged_data.loc[image.path] = row
        if i > 400: break
    tagged_data.to_csv(out_path)
    return adata


# testing and debuging
if __name__ == '__main__':
    dataset_name = '/FPST/data/Visium_Mouse_Olfactory_Bulb'
    data = st.Read10X(dataset_name)
    adata = tile_image(data, f'{dataset_name}/tiling/', crop_size=100)
    # generate_edge_detected_images(images_path = f'{dataset_name}/tiling/', out_path = f'{dataset_name}/', blur_kernel_size=(7,7))
    tag_images_for_edges(adata, edge_detect_images = f'{dataset_name}/tiling_classification_train/', out_path=f'{dataset_name}/adata.csv')