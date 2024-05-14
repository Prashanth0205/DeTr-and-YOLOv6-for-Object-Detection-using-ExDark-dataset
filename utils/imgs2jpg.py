
import os
from tqdm import tqdm

all_imgs = "../ExDark_All/Images"
all_annos = "../ExDark_All/Annotations"
imgs = "../ExDark/ExDark"
annos = "../ExDark_Annno/ExDark_Annno"

print("Converting all images to JPG...")

print(all_imgs)
for img in tqdm(os.listdir(all_imgs)):
    name = img.split(".")[0]
    format = img.split(".")[1]
    if format != "jpg":
        os.rename(f'{all_imgs}/{img}', f'{all_imgs}/{name}.jpg')

print(imgs)
for label_dir in tqdm(os.listdir(imgs)):
    for img in tqdm(os.listdir(f'{imgs}/{label_dir}')):
        name = img.split(".")[0]
        format = img.split(".")[1]
        if format != "jpg":
            os.rename(f'{imgs}/{label_dir}/{img}', f'{imgs}/{label_dir}/{name}.jpg')

print(all_annos)
for anno in tqdm(os.listdir(all_annos)):
    name = anno.split(".")[0]
    format = anno.split(".")[1]
    if format != "jpg":
        os.rename(f'{all_annos}/{anno}', f'{all_annos}/{name}.jpg.txt')

print(annos)
for label_dir in tqdm(os.listdir(annos)):
    for anno in tqdm(os.listdir(f'{annos}/{label_dir}')):
        name = anno.split(".")[0]
        format = anno.split(".")[1]
        if format != "jpg":
            os.rename(f'{annos}/{label_dir}/{anno}', f'{annos}/{label_dir}/{name}.jpg.txt')