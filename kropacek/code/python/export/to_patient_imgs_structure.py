import json
import os
import re
from pathlib import Path
from shutil import copyfile


if __name__ == '__main__':
    coloring = 'VG'
    working_dir = "F:/ondra/ateroskleroza-data/histology_segmentation/data/svetla_tmava/tmava/data"
    output_dir = "F:/ondra/ateroskleroza-data/histology_segmentation/data/patient_to_img"
    dirs = os.listdir(working_dir)
    with open('F:/ondra/ateroskleroza-data/histology_segmentation/misc/patient_images.json', 'r') as f:
        patient_images_descriptor = json.load(f)
    
    for dir_name in dirs:
        curr_dir = working_dir + "/" + dir_name
        masks = curr_dir + "/" + "masks/preprocessed"
        if not os.path.exists(masks):
            print("masks folder does not exist in: " + curr_dir)
            continue
        last_name_part = dir_name.split("_")[2].replace("x", "").zfill(4)
        original_img = curr_dir + '/' + dir_name + '_original.png'
        mask_colored = masks + '/all_masks_colored.png'
        mask_original = masks + '/all_masks_original.png'

        for patient_name in patient_images_descriptor.keys():
            patient_images = set(patient_images_descriptor[patient_name]['HEVG'])
            if last_name_part in patient_images:
                # multiple measurements for same patients with a/b/c at the end
                patient_name = re.sub(r'[a-zA-Z]', '', patient_name)
                patient_dir_name = output_dir + '/' + patient_name
                Path(patient_dir_name).mkdir(parents=True, exist_ok=True)
                if os.path.exists(mask_colored):
                    copyfile(original_img,
                            patient_dir_name + '/' + coloring + '_' + last_name_part + ".png")
                    copyfile(mask_colored,
                            patient_dir_name + '/' + coloring + '_' + last_name_part + "_label.png")
        
    print("run script done!")
    exit(0)


