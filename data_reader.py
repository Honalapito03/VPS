import cv2
import numpy as np
import os
import mapping
import pandas as pd

def load_image_as_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x, y = img.shape[0] // 2, img.shape[1] // 2
    size = min(x, y)
    img = img[x - size : x + size, y - size : y + size]
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return img


def read_image_dataset(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(directory, filename)
            img = load_image_as_grayscale(img_path)
            images.append((filename, img))
    return images

def read_ground_truth(file_path):
    gt_data = []
    df = pd.read_excel(file_path)
    df.columns = ['x', 'y', 'scale', 'rotation']

    for line in df.itertuples(index=False):
        parts = [line.x, line.y, line.scale, line.rotation]
        if len(parts) != 4:
            continue
        x, y, scale, rotation = parts
        gt_data.append({
            'x': float(x),
            'y': float(y),
            'scale': float(scale),
            'rotation': float(rotation)
        })
    return gt_data


def run_tests_on_dataset(image_dir, gt_file):
    images = read_image_dataset(image_dir)[1:-1]

    mapper = mapping.Mapper()
    mapper.current_image = images[0][1]
    mapper.resolution = max(images[0][1].shape)
    mapper.x_res = images[0][1].shape[1]
    mapper.y_res = images[0][1].shape[0]
    mapper.px_mm = 0.0625
    mapper.take_history = True

    mapper.start() 
    for (filename, img), gt in zip(images[1:], read_ground_truth(gt_file)):
        print(f"Processing image: {filename}")
        print(img)
        mapper.current_image = img
        mapper.loop_step()
        mapper.add_gt(gt["x"], gt["y"], gt["scale"], gt["rotation"])

    mapper.export_history(f"{image_dir}/test_results.xlsx")


if __name__ == "__main__":
    run_tests_on_dataset("Map_set_1", "Map_set_1/Dataset.xlsx")


    