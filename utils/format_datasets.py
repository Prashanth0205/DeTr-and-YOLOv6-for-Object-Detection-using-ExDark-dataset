import os
import cv2
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
from datasets_generators import CocoDetection
from transformers import DetrImageProcessor, DetrForObjectDetection
from image_enhancement_functions import clahe, color_balance_adjustment, histogram_equalization

def data2COCO():
    # Initialize COCO json format
    train_json = {
        "images": [],
        "annotations": [],
        "categories": [],
        }

    val_json = {
        "images": [],
        "annotations": [],
        "categories": [],
        }

    test_json = {
        "images": [],
        "annotations": [],
        "categories": [],
        }

    # Paths
    cat_dir = '../ExDark/ExDark'
    anno_dir = '../ExDark_All/Annotations'
    img_dir = '../ExDark_All/Images'
    coc_dir = '../ExDark_COCO'

    print('Generating labels...')

    # Categories (Subfolders)
    categories = os.listdir(cat_dir)
    for i, c in enumerate(categories):
        train_json["categories"].append({"id": i, 
                                        "name": c})
        val_json["categories"].append({"id": i,
                                        "name": c})
        test_json["categories"].append({"id": i,
                                        "name": c})
        
    # Path List
    img_list = os.listdir(img_dir)

    print('Splitting...')

    # Split
    train_list = []
    val_list = []
    test_list = []
    for subf in os.listdir(cat_dir):
        subdir = os.path.join(cat_dir, subf)
        imgs = os.listdir(subdir)
        train_list.extend(imgs[:250])
        val_list.extend(imgs[250:400])
        test_list.extend(imgs[400:])

    # Print values
    print('  Length of train set: ', len(train_list))
    print('  Length of val set: ', len(val_list))
    print('  Length of test set: ', len(test_list))

    print('Generating Training Set Json...')

    # Train Json
    img_id = 0
    anno_id = 0
    for img in tqdm(train_list):
        # Image
        train_json["images"].append({"id": img_id,
                                    "file_name": img})
        # Annotations
        with open(os.path.join(anno_dir, img + '.txt'), 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                train_json['annotations'].append({"id": anno_id,
                                                "image_id": img_id,
                                                "category_id": categories.index(l.split(' ')[0]),
                                                "bbox": l.split(' ')[1:5],
                                                "area": float(l.split(' ')[3])*float(l.split(' ')[4])})   
                # Update annotation id 
                anno_id += 1
        # Update image id
        img_id += 1

    print('Generating Validation Set Json...')

    # Val Json
    img_id = 0
    anno_id = 0
    for img in tqdm(val_list):
        # Image
        val_json["images"].append({"id": img_id,
                                "file_name": img})
        # Annotations
        with open(os.path.join(anno_dir, img + '.txt'), 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                val_json['annotations'].append({"id": anno_id,
                                                "image_id": img_id,
                                                "category_id": categories.index(l.split(' ')[0]),
                                                "bbox": l.split(' ')[1:5],
                                                "area": float(l.split(' ')[3])*float(l.split(' ')[4])})   
                # Update annotation id 
                anno_id += 1
        # Update image id
        img_id += 1

    print('Generating Test Set Json...')

    # Test Json
    img_id = 0
    anno_id = 0
    for img in tqdm(test_list):
        # Image
        test_json["images"].append({"id": img_id,
                                    "file_name": img})
        # Annotations
        with open(os.path.join(anno_dir, img + '.txt'), 'r') as f:
            lines = f.readlines()[1:]
            n_lines = len(lines)
            for l in lines:
                test_json['annotations'].append({"id": anno_id,
                                                "image_id": img_id,
                                                "category_id": categories.index(l.split(' ')[0]),
                                                "bbox": l.split(' ')[1:5],
                                                "area": float(l.split(' ')[3])*float(l.split(' ')[4]),
                                                "iscrowd": 1 if n_lines > 1 else 0})
                                                    
                # Update annotation id 
                anno_id += 1
        # Update image id
        img_id += 1

    print('Saving...')

    # Save Files
    json.dump(train_json, open(os.path.join(coc_dir, 'train_set.json'), 'w'))
    json.dump(val_json, open(os.path.join(coc_dir, 'val_set.json'), 'w'))
    json.dump(test_json, open(os.path.join(coc_dir, 'test_set.json'), 'w'))

    print('Done')

# -----------------------------------------------------------------------------------------------
    
def data2YOLO():

    # Main Folders
    imgs = "../ExDark/ExDark"
    annos = "../ExDark_Annno/ExDark_Annno"
    all_imgs = "../ExDark_All/Images"
    all_annos = "../ExDark_All/Annotations"

    print("Creating folders...")

    # Define folders to be created
    raw = "../ExDark_YOLO/RAW"
    raw_train = "../ExDark_YOLO/RAW/train"
    raw_train_images = "../ExDark_YOLO/RAW/train/images"
    raw_train_labels = "../ExDark_YOLO/RAW/train/labels"
    raw_valid = "../ExDark_YOLO/RAW/valid"
    raw_valid_images = "../ExDark_YOLO/RAW/valid/images"
    raw_valid_labels = "../ExDark_YOLO/RAW/valid/labels"
    clahe_p = "../ExDark_YOLO/CLAHE"
    clahe_train = "../ExDark_YOLO/CLAHE/train"
    clahe_train_images = "../ExDark_YOLO/CLAHE/train/images"
    clahe_train_labels = "../ExDark_YOLO/CLAHE/train/labels"
    clahe_valid = "../ExDark_YOLO/CLAHE/valid"
    clahe_valid_images = "../ExDark_YOLO/CLAHE/valid/images"
    clahe_valid_labels = "../ExDark_YOLO/CLAHE/valid/labels"
    cb = "../ExDark_YOLO/CB"
    cb_train = "../ExDark_YOLO/CB/train"
    cb_train_images = "../ExDark_YOLO/CB/train/images"
    cb_train_labels = "../ExDark_YOLO/CB/train/labels"
    cb_valid = "../ExDark_YOLO/CB/valid"
    cb_valid_images = "../ExDark_YOLO/CB/valid/images"
    cb_valid_labels = "../ExDark_YOLO/CB/valid/labels"
    he = "../ExDark_YOLO/HE"
    he_train = "../ExDark_YOLO/HE/train"
    he_train_images = "../ExDark_YOLO/HE/train/images"
    he_train_labels = "../ExDark_YOLO/HE/train/labels"
    he_valid = "../ExDark_YOLO/HE/valid"
    he_valid_images = "../ExDark_YOLO/HE/valid/images"
    he_valid_labels = "../ExDark_YOLO/HE/valid/labels"
    test = "../ExDark_YOLO/test"
    test_images = "../ExDark_YOLO/test/images"
    test_labels = "../ExDark_YOLO/test/labels"

    # Create folders
    os.makedirs(raw, exist_ok=True)
    os.makedirs(raw_train, exist_ok=True)
    os.makedirs(raw_train_images, exist_ok=True)
    os.makedirs(raw_train_labels, exist_ok=True)
    os.makedirs(raw_valid, exist_ok=True)
    os.makedirs(raw_valid_images, exist_ok=True)
    os.makedirs(raw_valid_labels, exist_ok=True)
    os.makedirs(clahe_p, exist_ok=True)
    os.makedirs(clahe_train, exist_ok=True)
    os.makedirs(clahe_train_images, exist_ok=True)
    os.makedirs(clahe_train_labels, exist_ok=True)
    os.makedirs(clahe_valid, exist_ok=True)
    os.makedirs(clahe_valid_images, exist_ok=True)
    os.makedirs(clahe_valid_labels, exist_ok=True)
    os.makedirs(cb, exist_ok=True)
    os.makedirs(cb_train, exist_ok=True)
    os.makedirs(cb_train_images, exist_ok=True)
    os.makedirs(cb_train_labels, exist_ok=True)
    os.makedirs(cb_valid, exist_ok=True)
    os.makedirs(cb_valid_images, exist_ok=True)
    os.makedirs(cb_valid_labels, exist_ok=True)
    os.makedirs(he, exist_ok=True)
    os.makedirs(he_train, exist_ok=True)
    os.makedirs(he_train_images, exist_ok=True)
    os.makedirs(he_train_labels, exist_ok=True)
    os.makedirs(he_valid, exist_ok=True)
    os.makedirs(he_valid_images, exist_ok=True)
    os.makedirs(he_valid_labels, exist_ok=True)
    os.makedirs(test, exist_ok=True)
    os.makedirs(test_images, exist_ok=True)
    os.makedirs(test_labels, exist_ok=True)

    print("Splitting data...")

    # Split data
    train_data = []
    valid_data = []
    test_data = []

    for label_dir in os.listdir(imgs):
        train_data.extend(os.listdir(os.path.join(imgs, label_dir))[:250])
        valid_data.extend(os.listdir(os.path.join(imgs, label_dir))[250:400])
        test_data.extend(os.listdir(os.path.join(imgs, label_dir))[400:])

    print(f'  Split: {len(train_data)} - {len(valid_data)} - {len(test_data)}')

    print("Defining labels...")

    # Labels
    labels = dict([(label, i) for i, label in enumerate(os.listdir(imgs))])
    print(f'  Labels: {labels}')

    print("Applying enhancements and copying images...")

    # Copy images to folders
    for img in tqdm(os.listdir(all_imgs)):
        
        # Read RAW image
        image = cv2.imread(os.path.join(all_imgs, img))

        # Train
        if img in train_data:
            # RAW
            shutil.copy(os.path.join(all_imgs, img), os.path.join(raw_train_images, img))

            # CLAHE
            clahe_image = clahe(image)
            cv2.imwrite(os.path.join(clahe_train_images, img), clahe_image)

            # CB
            cb_image = color_balance_adjustment(image)
            cv2.imwrite(os.path.join(cb_train_images, img), cb_image)

            # HE
            he_image = histogram_equalization(image)
            cv2.imwrite(os.path.join(he_train_images, img), he_image)

        # Valid
        elif img in valid_data:
            # RAW
            shutil.copy(os.path.join(all_imgs, img), os.path.join(raw_valid_images, img))

            # CLAHE
            clahe_image = clahe(image)
            cv2.imwrite(os.path.join(clahe_valid_images, img), clahe_image)

            # CB
            cb_image = color_balance_adjustment(image)
            cv2.imwrite(os.path.join(cb_valid_images, img), cb_image)

            # HE
            he_image = histogram_equalization(image)
            cv2.imwrite(os.path.join(he_valid_images, img), he_image)

        # Test
        else:
            shutil.copy(os.path.join(all_imgs, img), os.path.join(test_images, img))

    print("Converting all annotations to TXT...")
    for ann in os.listdir(all_annos):
        file = f"{all_annos}/{ann}"
        raw_file = open(file, "r")
        raw_cont = raw_file.read().strip().split("\n")[1:]
        new_cont = ""
        for i in raw_cont:
            obj_list = i.split(" ")[:5]
            obj_list[0] = labels[obj_list[0]]

            # Normalizing part
            bbox = np.array(obj_list[1:]).astype(float)
            bbox = np.round([float(i)/np.sum(bbox) for i in bbox], 3)
                    
            obj_list[1:] = bbox
            new_cont += " ".join(map(str, obj_list)) + "\n"

            # Write new content in a new file
            

    print("Copying labels...")

    # Copy labels to folders
    for anno in tqdm(os.listdir(all_annos)):
        # Train
        if anno in train_data:
            shutil.copy(os.path.join(all_annos, anno), os.path.join(raw_train_labels, anno))
            shutil.copy(os.path.join(all_annos, anno), os.path.join(clahe_train_labels, anno))
            shutil.copy(os.path.join(all_annos, anno), os.path.join(cb_train_labels, anno))
            shutil.copy(os.path.join(all_annos, anno), os.path.join(he_train_labels, anno))

        # Valid
        elif anno in valid_data:
            shutil.copy(os.path.join(all_annos, anno), os.path.join(raw_valid_labels, anno))
            shutil.copy(os.path.join(all_annos, anno), os.path.join(clahe_valid_labels, anno))
            shutil.copy(os.path.join(all_annos, anno), os.path.join(cb_valid_labels, anno))
            shutil.copy(os.path.join(all_annos, anno), os.path.join(he_valid_labels, anno))

        # Test
        else:
            shutil.copy(os.path.join(all_annos, anno), os.path.join(test_labels, anno))

    print("Creating YAML files...")

    # Create YAML files
    with open("../ExDark_YOLO/raw_train.yaml", "w") as f:
        f.write(f"train: {raw_train}\n")
        f.write(f"val: {raw_valid}\n")
        f.write(f"nc: {len(labels)}\n")
        f.write(f"names: {list(labels.keys())}")

    with open("../ExDark_YOLO/clahe_train.yaml", "w") as f:
        f.write(f"train: {clahe_train}\n")
        f.write(f"val: {clahe_valid}\n")
        f.write(f"nc: {len(labels)}\n")
        f.write(f"names: {list(labels.keys())}")

    with open("../ExDark_YOLO/cb_train.yaml", "w") as f:
        f.write(f"train: {cb_train}\n")
        f.write(f"val: {cb_valid}\n")
        f.write(f"nc: {len(labels)}\n")
        f.write(f"names: {list(labels.keys())}")

    with open("../ExDark_YOLO/he_train.yaml", "w") as f:
        f.write(f"train: {he_train}\n")
        f.write(f"val: {he_valid}\n")
        f.write(f"nc: {len(labels)}\n")
        f.write(f"names: {list(labels.keys())}")

    print("Done!")

# -----------------------------------------------------------------------------------------------

def dert_results2COCO():
    name = None
    enhancement = None
    inp = input("Enter enhancement:\n  1. Clahe\n  2. Color Balance Adjustement\n  3. Histogram Equialization\n  4. No Enhancement\n>")
    if inp == "1":
        name = "clahe/output"
        enhancement = "clahe"
    elif inp == "2":
        name = "color_balance_adjustment/output"
        enhancement = "color_balance_adjustment"
    elif inp == "3":
        name = "he/output"
        enhancement = "he"
    elif inp == "4":
        name = "raw/output"
        enhancement = 'raw'
    else:
        print("Invalid input")
        return

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    imgs_path = "../ExDark_All/Images"
    test_path = "../ExDark_COCO/test_set.json"
    model_path = f"../Models/Transformer/lightning_logs/{name}"

    # Initialization
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    test_set = CocoDetection(image_directory_path=imgs_path,  annotation_file_path=test_path, image_processor=image_processor, enhancement=enhancement)
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    results = {"annotations": []}
    
    # Loop
    image_id = 0
    for i in tqdm(range(len(test_set))):

        # Original Image
        _, anno = test_set[i]
        file_name = test_set.coco.loadImgs(i)[0]['file_name']
        original_size = anno['orig_size']
        cv_image = cv2.imread(os.path.join(imgs_path, file_name))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Prediction
        results_ = None
        with torch.no_grad():
            inputs = image_processor(images=cv_image, return_tensors='pt').to(device)
            outputs = model(**inputs)
            results_ = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=0.001, 
                target_sizes=[original_size]
            )[0]

        for j in range(len(results_["scores"])):
            if results_["scores"][j] >= 0.85:
                bbox = results_["boxes"][j].tolist()
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                bbox = [float(b) for b in bbox]
                results['annotations'].append({"image_id": image_id,
                                               "category_id": results_["labels"][j].item(),
                                               "bbox": bbox,
                                               "score": results_["scores"][j].item(),})	
        image_id += 1

    # Save as json
    with open(f'../Models/Transformer/lightning_logs/{name}/results.json', 'w') as fp:
        json.dump(results, fp)

# -----------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    function = input("Select function:\n  1 - data2COCO\n  2 - data2YOLO\n  3 - dert_results2COCO\n>")
    if function == "1":
        data2COCO()
    elif function == "2":
        data2YOLO()
    elif function == "3":
        dert_results2COCO()
    else:
        print("Invalid function")