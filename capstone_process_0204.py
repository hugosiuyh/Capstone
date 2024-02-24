import os
import json
import shutil
from PIL import Image, ExifTags
import labelme2coco
import random
from collections import defaultdict

# Base directory where the folders are stored

base_dir = '/Users/hugo/Downloads/影像集_noaphidandthrip_updated'

# New base directory for rearranged dataset
rearranged_base_dir = f'{base_dir}_rearranged'
os.makedirs(rearranged_base_dir, exist_ok=True)

final_base_dir = f'{base_dir}_final'
os.makedirs(final_base_dir, exist_ok=True)

# Store all files from all diseases
all_files = []

no_annotations_dir = os.path.join(final_base_dir, 'no_annotations')
os.makedirs(no_annotations_dir, exist_ok=True)

# Iterate over disease folders
for disease in os.listdir(base_dir):
    disease_path = os.path.join(base_dir, disease)

    # Skip if it's not a directory
    if not os.path.isdir(disease_path):
        continue

    # Iterate over plant folders
    for plant in os.listdir(disease_path):
        plant_path = os.path.join(disease_path, plant)

        # Check if the plant path is a directory
        if os.path.isdir(plant_path):
            # Iterate over files in plant folder
            for file in os.listdir(plant_path):
                if file.endswith('.JPG'):
                    image_path = os.path.join(plant_path, file)
                    json_name = file.replace('.JPG', '.json')
                    json_path = os.path.join(plant_path, json_name)

                    # Check if corresponding JSON file exists
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            if not data['shapes']:  # Assuming 'shapes' field holds the annotations
                                # Move files to no_annotations folder
                                os.remove(json_path)
                                shutil.copy(image_path, os.path.join(no_annotations_dir, file))
                            else:
                                all_files.append((image_path, json_path))
                    else:
                        # Move image file to no_annotations folder if JSON does not exist
                        shutil.copy(image_path, os.path.join(no_annotations_dir, file))


# Process and move files
for image_path, json_path in all_files:
    
    # Modify JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Modify labels in JSON data
    for annotation in data['shapes']:
        label = annotation['label']
        disease, plant = label.split('_')[0], label.split('_')[1]

        # Apply the specific conditions for label modification
        if disease[:5] == "aphid":
            annotation['label'] == 'aphid'
        elif len(plant) > 3:
            if plant[-1] == 'b' and disease != 'dmildw':
                annotation['label'] = f'{disease}_b'
            elif plant[-1] == 'l' or disease == 'dmildw':
                annotation['label'] = f'{disease}_l'
        elif len(plant) == 3:
            annotation['label'] = f'{disease}_b'

    # Write the modified JSON back
    new_json_path = os.path.join(rearranged_base_dir, os.path.basename(json_path))
    with open(new_json_path, 'w') as f:
        json.dump(data, f)

    # Move the image file
    shutil.copy(image_path, os.path.join(rearranged_base_dir, os.path.basename(image_path)))
    
    
print('Finished label-processing and relocating files.')

def resize_image_and_annotations(dir, max_length):
    # Iterate over files directly in the base directory
    for file in os.listdir(dir):
        if file.endswith('.json'):
            json_path = os.path.join(dir, file)
            image_path = json_path.replace('.json', '.JPG')  # Adjust the extension as necessary

            #Check if the image needs rotation
            rotation_needed = needs_rotation(image_path)
            if rotation_needed is not None:
                with open(json_path, 'r') as f:
                    img_data = json.load(f)
                print('old',img_data['shapes'])
                annotations = img_data['shapes']  # Modify this based on your JSON structure
                new_img_data = rotate_image(image_path, img_data, annotations, rotation_needed)

                # Update annotations in JSON
    
                img_data['shapes'] = new_img_data["shapes"]
                print('new',img_data['shapes'])
                
                with open(json_path, 'w') as f:
                    json.dump(img_data, f, indent=4)

            # Resize the image
            with Image.open(image_path) as img:
                original_size = img.size
                scale_factor = min(max_length / max(original_size), 1.0)  # No upscaling
                new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
                img = img.resize(new_size, Image.ANTIALIAS)
                img.save(image_path)
            
            # Resize annotations in the JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for shape in data['shapes']:
                for point in shape['points']:
                    point[0] *= scale_factor
                    point[1] *= scale_factor
                    
            data['imageWidth'] *= scale_factor
            data['imageHeight'] *= scale_factor
                
            # Overwrite the original JSON file with modified annotations
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            new_image_path = os.path.join(final_base_dir, os.path.basename(image_path))
            shutil.move(image_path, new_image_path)

# Continue with the rest of your functions (needs_rotation, rotate_point, rotate_image)
def needs_rotation(image_path):
    try:
        with Image.open(image_path) as image:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()

            # Orientation values and corresponding rotations in degrees
            orientation_mapping = {
                3: 180,  # Upside down
                6: 90,  # 90 CW
                8: 270    # 90 CCW
            }

            return orientation_mapping.get(exif[orientation], None)
    except (AttributeError, KeyError, IndexError):
        return None

def rotate_image(image_path, img_data, annotations, degrees):
    with Image.open(image_path) as image:
        original_width, original_height = image.size
        exif = image.info.get('exif', b'')  # Extract the original EXIF data
        rotated = image.rotate(-degrees, expand=True)  # Negative because PIL's rotate is counter-clockwise
        rotated.save(image_path, exif=exif)  # Pass the EXIF data back when saving
        print(f"Rotated {degrees} degrees and saved image: {image_path}")
        return img_data

max_length = 800
resize_image_and_annotations(rearranged_base_dir, max_length)

# set directory that contains labelme annotations and image files
labelme_folder = rearranged_base_dir #new_base_dir

# set train split rate
train_split_rate = 0.7

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, final_base_dir, train_split_rate)

import json
import random
from collections import defaultdict

def balanced_split_coco_dataset(json_file_path, output_path1, output_path2):
    # Load the COCO dataset JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Group annotations by image
    image_annotations = defaultdict(list)
    for annotation in data['annotations']:
        image_annotations[annotation['image_id']].append(annotation)

    # Create lists of image ids and shuffle them
    image_ids = list(image_annotations.keys())
    random.shuffle(image_ids)

    # Split the image ids in half
    mid_index = len(image_ids) // 2
    image_ids1 = set(image_ids[:mid_index])
    image_ids2 = set(image_ids[mid_index:])

    # Split images and annotations based on the split image IDs
    images1, images2 = [], []
    annotations1, annotations2 = [], []
    new_image_id, new_annotation_id = 1, 1

    for image in data['images']:
        if image['id'] in image_ids1:
            image['id'] = new_image_id
            images1.append(image)
            for annotation in image_annotations[image['id']]:
                annotation['id'] = new_annotation_id
                annotation['image_id'] = new_image_id
                annotations1.append(annotation)
                new_annotation_id += 1
            new_image_id += 1
        elif image['id'] in image_ids2:
            image['id'] = new_image_id
            images2.append(image)
            for annotation in image_annotations[image['id']]:
                annotation['id'] = new_annotation_id
                annotation['image_id'] = new_image_id
                annotations2.append(annotation)
                new_annotation_id += 1
            new_image_id += 1

    # Create two new datasets
    data1 = {
        'images': images1,
        'annotations': annotations1,
        'categories': data['categories']
    }

    data2 = {
        'images': images2,
        'annotations': annotations2,
        'categories': data['categories']
    }

    # Save the new datasets
    with open(output_path1, 'w') as file:
        json.dump(data1, file)

    with open(output_path2, 'w') as file:
        json.dump(data2, file)

    return output_path1, output_path2

# Example usage
val_dir = f'{final_base_dir}/val.json'
test_dir = f'{final_base_dir}/test.json'
balanced_split_coco_dataset(val_dir, val_dir, test_dir)
