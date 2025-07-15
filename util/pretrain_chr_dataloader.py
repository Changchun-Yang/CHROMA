import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import json
import torchvision.transforms.functional as TF
import math
import random
import numpy as np
from torchvision import transforms

class ChrTransform:
    def __init__(self, output_size):
        self.output_size = output_size  

    def __call__(self, image, bbox=None, w_coljit=False, w_flip=False, w_rotate=False, w_scale=False, w_randp=False, pad_color=(0,0,0)):

        if bbox:
            x, y, w, h = bbox
            chromosome_region = image.crop((x, y, x + w, y + h))
        else:
            chromosome_region = image

        if w_coljit:
            color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5)
            chromosome_region = color_jitter(chromosome_region)

        if w_scale:
            chromosome_region = self.resize_scale(chromosome_region)

        if w_flip:
            chromosome_region = self.apply_flip(chromosome_region)
        if w_rotate:
            chromosome_region = self.apply_rotate(chromosome_region)
        
        
        chromosome_region, new_size = self.maintain_aspect_ratio(chromosome_region)

        new_image = Image.new('RGB', self.output_size, pad_color)
        if w_randp:
            max_x = self.output_size[0] - new_size[0]
            max_y = self.output_size[1] - new_size[1]
            upper_left_x = random.randint(0, max_x) if max_x > 0 else 0
            upper_left_y = random.randint(0, max_y) if max_y > 0 else 0
            new_image.paste(chromosome_region, (upper_left_x, upper_left_y))
            new_bbox = [upper_left_x, upper_left_y, new_size[0], new_size[1]]
        else:
            upper_left_x = (self.output_size[0] - new_size[0]) // 2
            upper_left_y = (self.output_size[1] - new_size[1]) // 2
            new_image.paste(chromosome_region, (upper_left_x, upper_left_y))

            new_bbox = [upper_left_x, upper_left_y, new_size[0], new_size[1]]
        return new_image, new_bbox

    def apply_flip(self, region):
        # print(region.size)
        if torch.rand(1) < 0.5:
            region = TF.hflip(region)
        if torch.rand(1) < 0.5:
            region = TF.vflip(region)
        return region
    
    def __get_rotated_corners(self, w, h, angle):
        angle_rad = math.radians(angle)
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        new_corners = []
        cx, cy = w / 2, h / 2  
        for x, y in corners:
            x -= cx
            y -= cy
            new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            new_x += cx
            new_y += cy
            new_corners.append((new_x, new_y))
        return new_corners
    
    def __get_bounding_box(self, corners):
        min_x = min(corner[0] for corner in corners)
        max_x = max(corner[0] for corner in corners)
        min_y = min(corner[1] for corner in corners)
        max_y = max(corner[1] for corner in corners)
        return (min_x, min_y, max_x, max_y)

    def apply_rotate(self, region):
        angle = torch.randint(-60, 61, (1,)).item()
        # rotate_region = TF.rotate(region, angle, expand=True)
        rotate_region = region.rotate(angle, expand=True)
        # directly using the TF.rotate will increase the size
        # w, h = region.size
        # corners = self.__get_rotated_corners(w, h, angle)
        # min_x, min_y, max_x, max_y = self.__get_bounding_box(corners)
        bbox = rotate_region.getbbox()

        cropped_region = rotate_region.crop(bbox)
        return cropped_region

    def resize_scale(self, region):
        # ori_size = region.size
        # def get_scale(ratio):
        #     random_value = torch.rand(1).item()
        #     log_scale = ratio * random_value - ratio/2.0 
        #     scale = 2 ** log_scale  
        #     return scale
        # scale_range = 4
        # scale = get_scale(scale_range)
        # scale_range = (0.8, 1.2)
        # scale = random.uniform(*self.scale_range)
        # new_size = (int(ori_size[0] * scale), int(ori_size[1] * scale))
        # PIL.Image.resize() use (width, height) 
        # however TF.resize() use (height, width)
        # region = TF.resize(region, (new_size[1], new_size[0]))
        # region = region.resize(new_size)

        width, height = region.size

        scale_range = (0.25, 1.2)
        scale = random.uniform(*scale_range)
        # print(scale)

        # def get_scale(ratio):
        #     random_value = torch.rand(1).item()
        #     log_scale = ratio * random_value - ratio/2.0 
        #     scale = 2 ** log_scale  
        #     return scale
        # scale_range = 2
        # scale = get_scale(scale_range)

        target_height = int(height * scale)
        target_width = int(width * scale)

        # If scale is greater than 1, enlarge the region
        if scale > 1:
            # Resize the region to the larger size
            resized_region = TF.resize(region, [target_height, target_width], interpolation=TF.InterpolationMode.BILINEAR)
            return resized_region
        else:
            # Compute top-left corner of the cropped area for a random crop
            crop_top = random.randint(0, height - target_height)
            crop_left = random.randint(0, width - target_width)

            # Crop and then resize back to the original size
            cropped_resized_region = TF.resized_crop(
                region, crop_top, crop_left, target_height, target_width,
                size=[height, width],  # Resize back to the original size
                interpolation=TF.InterpolationMode.BICUBIC
            )
            return cropped_resized_region
        

    def maintain_aspect_ratio(self, region):
        ### Prevent out of bounds, scale according to aspect ratio
        new_size = region.size
        # print(new_size)
        ratio_width = self.output_size[0] / new_size[0]
        ratio_height = self.output_size[1] / new_size[1]
        ratio = min(ratio_width, ratio_height)
        if new_size[0] > self.output_size[0] or new_size[1] > self.output_size[1]:
            new_size = (int(new_size[0] * ratio), int(new_size[1] * ratio))
            # print(new_size)
            # PIL.Image.resize() use (width, height) 
            # however TF.resize() use (height, width)
            # region = TF.resize(region, (new_size[1], new_size[0]))
            region = region.resize(new_size)
        return region, new_size


def adjust_bbox(image, bbox, divisor=16):
    img_width, img_height = image.size
    
    x, y, w, h = bbox
    
    new_x = (x // divisor) * divisor
    new_y = (y // divisor) * divisor
    new_w = ((x + w + divisor - 1) // divisor) * divisor - new_x
    new_h = ((y + h + divisor - 1) // divisor) * divisor - new_y
    
    new_w = min(new_w, img_width - new_x)
    new_h = min(new_h, img_height - new_y)
    
    new_w = max(0, new_w)
    new_h = max(0, new_h)
    
    return [new_x, new_y, new_w, new_h]


class ChrDataset(Dataset):
    # chr dataset with bbox or mask, metedata files, every chromosome located on the center of the image
    def __init__(self, dataset_dir, data_size=(224,224), patch_size=16, transform=True, w_coljit=True, w_flip=True, w_rotate=True, w_scale=False, w_randp=True, pad_color=(0,0,0), normalize=False):
        """
        Args:
            dataset_dir (string): Root directory of the dataset containing images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            w_coljit: whether to apply color jitter
            w_flip: whether to apply flip
            w_rotate: whether to apply rotate
            w_scale: whether to apply scale
            w_randp: whether to apply random paste
            pad_color: padding to light or dark
            normalize (dict or None): Normalization parameters. If None, no normalization is applied.
                                      If dict, it should have 'mean' and 'std' keys with lists as values.
        """
        self.dataset_dir = dataset_dir
        self.metadata_dir = os.path.join(dataset_dir, 'metadata')
        self.transform = transform
        self.w_coljit = w_coljit
        self.w_flip = w_flip
        self.w_rotate = w_rotate
        self.w_scale = w_scale
        self.w_randp = w_randp
        self.pad_color = pad_color
        self.data_size = data_size
        self.patch_size = patch_size
        self.normalize = normalize
        self.metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.metadata_files)

    def __getitem__(self, idx):
        # Get the path to the JSON file for the current index
        metadata_file = os.path.join(self.metadata_dir, self.metadata_files[idx])
        
        # Read the JSON file
        with open(metadata_file, 'r') as f:
            img_info = json.load(f)

        # Construct the full path for the image and the mask
        img_path = os.path.join(self.dataset_dir, img_info['image'])
        mask_path = os.path.join(self.dataset_dir, img_info['mask'])

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        # mask = Image.open(mask_path).convert('L')  # Assuming mask is in grayscale

        # x,y,w,h = img_info['bbox']
        # bbox = (x, y, x + w, y + h)
        bbox = img_info['bbox']
        # Apply transformations, if any
        if self.transform:
            chr_transform = ChrTransform(self.data_size)
            image, bbox = chr_transform(image, bbox, self.w_coljit, self.w_flip, self.w_rotate, self.w_scale, self.w_randp, self.pad_color)
            # mask = self.transform(mask)  # Assuming same transform is applicable for mask
        
         # Convert image to tensor and scale to [0, 1]
        bbox = adjust_bbox(image, bbox)

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        if self.normalize:
            normalization_transform = transforms.Normalize(mean=self.normalize['mean'], std=self.normalize['std'])
            image = normalization_transform(image)

        bbox = torch.tensor(bbox)
        sample = {'image': image, 'bbox': bbox}

        return sample
    

class ChrDataset2(Dataset):
    # chr dataset only one single chr , the size of image==bbox
    def __init__(self, dataset_dir, data_size=(224,224), patch_size=16, transform=True, w_coljit=True, w_flip=True, w_rotate=True, w_scale=False, w_randp=True, pad_color=(0,0,0), normalize=False):
        """
        Args:
            dataset_dir (string): Root directory of the dataset containing images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            w_coljit: whether to apply color jitter
            w_flip: whether to apply flip
            w_rotate: whether to apply rotate
            w_scale: whether to apply scale
            w_randp: whether to apply random paste
            pad_color: padding the size to light or dark for background
            normalize (dict or None): Normalization parameters. If None, no normalization is applied.
                                      If dict, it should have 'mean' and 'std' keys with lists as values.
        """
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.transform = transform
        self.w_coljit = w_coljit
        self.w_flip = w_flip
        self.w_rotate = w_rotate
        self.w_scale = w_scale
        self.w_randp = w_randp
        self.pad_color = pad_color
        self.data_size = data_size
        self.patch_size = patch_size
        self.normalize = normalize
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        # mask = Image.open(mask_path).convert('L')  # Assuming mask is in grayscale

        # x,y,w,h = img_info['bbox']
        # bbox = (x, y, x + w, y + h)
        bbox = None
        # Apply transformations, if any
        if self.transform:
            chr_transform = ChrTransform(self.data_size)
            image, bbox = chr_transform(image, bbox, self.w_coljit, self.w_flip, self.w_rotate, self.w_scale, self.w_randp, self.pad_color)
            # mask = self.transform(mask)  # Assuming same transform is applicable for mask
        
        # Convert bbox size to Integer multiple of patch size
        bbox = adjust_bbox(image, bbox)
        # Convert image to tensor and scale to [0, 1]
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        if self.normalize:
            normalization_transform = transforms.Normalize(mean=self.normalize['mean'], std=self.normalize['std'])
            image = normalization_transform(image)

        bbox = torch.tensor(bbox)
        sample = {'image': image, 'bbox': bbox}

        return sample
    

if __name__ == '__main__':

    import glob
    def check_image_type(image):
        if image.mode == 'RGB':
            return "rgb"
        elif image.mode == 'L':
            return "l"
        else:
            return "other"
    def check_channels_identical(image):
        if image.mode != 'RGB':
            return "not rgb"
        
        data = np.array(image)
        
        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
        
        if np.array_equal(red, green) and np.array_equal(green, blue):
            return "equal"
        else:
            return "not equal"
    
    chr_transform = ChrTransform((224,224))
    w_coljit=   False
    w_flip=     True
    w_rotate=   False
    w_scale=    True
    w_randp=    True
    data_path = '/ibex/project/c2277/data/Karyotype/pretrain_processed'

    # # test dataset private
    # private_data_path = os.path.join(data_path, 'private')
    # metadata_file_path = os.path.join(private_data_path, 'metadata')
    
    # for metadata_file in glob.glob(f'{metadata_file_path}/**/*', recursive=True):
    #     image_filename = metadata_file.split('/')[-1]
    #     # Read the JSON file
    #     with open(metadata_file, 'r') as f:
    #         img_info = json.load(f)

    #     # Construct the full path for the image and the mask
    #     img_path = os.path.join(private_data_path, img_info['image'])
    #     mask_path = os.path.join(private_data_path, img_info['mask'])

    #     # Load image and mask
    #     image = Image.open(img_path).convert('RGB')
    #     image_array = np.array(image)
    #     # image = Image.open(img_path)
    #     print(image_array.shape)
    #     print(check_image_type(image))
    #     print(check_channels_identical(image))
    #     bbox = img_info['bbox']
    #     image, bbox = chr_transform(image, bbox)
    #     bbox = adjust_bbox(image, bbox)
    #     x,y,w,h = bbox
    #     draw = ImageDraw.Draw(image)
    #     draw.rectangle((x,y,x+w,y+h), outline='red', width=2)

    #     # Save the result
    #     output_path = os.path.join(data_path, 'tmp_private/' + image_filename + '.png')
    #     image.save(output_path)

    private_data_path = os.path.join(data_path, 'private_2')
    private_image_dir = os.path.join(private_data_path, 'images')
    
    for image_file in glob.glob(f'{private_image_dir}/**/*', recursive=True):
        image_filename = image_file.split('/')[-1]

        img_path = os.path.join(private_image_dir, image_filename)

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        image_array = np.array(image)
        # image = Image.open(img_path)
        print(image_array.shape)
        print(check_image_type(image))
        print(check_channels_identical(image))
        bbox = None
        image, bbox = chr_transform(image, bbox, w_coljit, w_flip, w_rotate, w_scale, w_randp)
        bbox = adjust_bbox(image, bbox)
        x,y,w,h = bbox
        draw = ImageDraw.Draw(image)
        draw.rectangle((x,y,x+w,y+h), outline='red', width=2)

        # Save the result
        output_path = os.path.join(data_path, 'tmp_private/' + image_filename)
        image.save(output_path)
