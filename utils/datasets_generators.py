import torch
import numpy as np
import imgaug as ia
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.datasets import CocoDetection
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from image_enhancement_functions import clahe, color_balance_adjustment, histogram_equalization

class CocoDetection(CocoDetection):
    def __init__(
        self, 
        annotation_file_path: str,
        image_directory_path: str,
        enhancement: str,
        image_processor, 
        train: bool = True,
    ):
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor
        self.enhancement = enhancement

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations2 = {'image_id': image_id, 'annotations': annotations}
        
        ''' FACEBOOK PREPROCESSING '''
        encoding = self.image_processor(images=images, annotations=annotations2, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        

        ''' CUSTOM PREPROCESSING '''
        # Image
        my_pixel_values = np.array(images)
        # Enhance
        if self.enhancement == 'clahe':
            my_pixel_values = clahe(my_pixel_values)
        elif self.enhancement == 'color_balance_adjustment':
            my_pixel_values = color_balance_adjustment(my_pixel_values)
        elif self.enhancement == 'he':
            my_pixel_values = histogram_equalization(my_pixel_values)
        else:
            pass
        # Resize
        size1, size2 = 700, 700
        image_rescaled = ia.imresize_single_image(my_pixel_values, (size1, size2))
        image_rescaled_Tensor = ToTensor()(image_rescaled)

        # Target
        my_target = {'size': torch.from_numpy(np.array([image_rescaled_Tensor.shape[1], image_rescaled_Tensor.shape[2]])).to(torch.int64),
                     'image_id': torch.from_numpy(np.array([image_id])).to(torch.int64),
        }

        # Annotations
        class_labels = []
        bboxes = []
        areas = []
        for ann in annotations:
            # Category
            class_labels.append(ann['category_id'])
            
            # Bounding Box
            og_box = [float(b) for b in ann['bbox']]
            bbs = BoundingBoxesOnImage([BoundingBox(x1=og_box[0], y1=og_box[1], x2=og_box[0]+og_box[2], y2=og_box[1]+og_box[3])],
                                        shape=my_pixel_values.shape)
            
            # Rescaled Bounding Box
            bbs_rescaled = bbs.on(image_rescaled)
            new_x1 = round(bbs_rescaled[0].x1)
            new_y1 = round(bbs_rescaled[0].y1)
            new_w = round(bbs_rescaled[0].x2) - round(bbs_rescaled[0].x1)
            new_h = round(bbs_rescaled[0].y2) - round(bbs_rescaled[0].y1)

            # Get cerner of the bounding box
            new_center_x = new_x1 + new_w/2
            new_center_y = new_y1 + new_h/2

            # Normalized Rescaled Bounding Box
            n_new_center_x = new_center_x / size2
            n_new_center_y = new_center_y / size1
            n_new_w = new_w / size2
            n_new_h = new_h / size1

            bboxes.append([n_new_center_x, n_new_center_y, n_new_w, n_new_h])
            areas.append(new_w * new_h)

        # Other annotations
        my_target['class_labels'] = torch.from_numpy(np.array(class_labels)).to(torch.int64)
        my_target['boxes'] = torch.from_numpy(np.array(bboxes)).to(torch.float32)
        my_target['area'] = torch.from_numpy(np.array(areas)).to(torch.float32)
        my_target['iscrowd'] = torch.zeros((len(annotations),), dtype=torch.int64)
        my_target['orig_size'] = torch.from_numpy(np.array([my_pixel_values.shape[0], my_pixel_values.shape[1]])).to(torch.int64)

        return image_rescaled_Tensor, my_target #, pixel_values, target