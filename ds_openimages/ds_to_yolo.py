import pandas as pd

df = pd.read_csv("ds_openimages/validation/labels/detections.csv")

for ind in df.index:
    f = open(f"ds_openimages/yolo/labels/validation/{df['ImageID'][ind]}.txt",'a')

    centerX = (df['XMax'][ind]+df['XMin'][ind])/2
    centerY = (df['YMax'][ind]+df['YMin'][ind])/2

    width = (df['XMax'][ind]-df['XMin'][ind])
    heigth = (df['YMax'][ind]-df['YMin'][ind])
    f.write(f"0 {centerX} {centerY} {width} {heigth}\n")
    f.close()

df = pd.read_csv("ds_openimages/train/labels/detections.csv")

for ind in df.index:
    f = open(f"ds_openimages/yolo/labels/train/{df['ImageID'][ind]}.txt",'a')

    centerX = (df['XMax'][ind]+df['XMin'][ind])/2
    centerY = (df['YMax'][ind]+df['YMin'][ind])/2

    width = (df['XMax'][ind]-df['XMin'][ind])
    heigth = (df['YMax'][ind]-df['YMin'][ind])
    f.write(f"0 {centerX} {centerY} {width} {heigth}\n")
    f.close()
    