import json
from .coco_classes import COCO_CATEGORIES, TEC_COCO_CATEGORIES


annFiles = [
    '../localDatasets/SMALL_Detection_Tiles_Dataset-COCO_CAT-imgsize_1920-tilesize_728-classes_bird-bgsample_2.0/train/_annotations.coco.json',
    '../localDatasets/SMALL_Detection_Tiles_Dataset-COCO_CAT-imgsize_1920-tilesize_728-classes_bird-bgsample_2.0/valid/_annotations.coco.json',
    '../localDatasets/SMALL_Detection_Tiles_Dataset-COCO_CAT-imgsize_1920-tilesize_728-classes_bird-bgsample_2.0/test/_annotations.coco.json',

]
for annFile in annFiles:
    print(f"Replacing COCO categories in annotation file: {annFile}")
    with open(annFile) as f:
        d = json.load(f)

    d['categories'] = COCO_CATEGORIES
        
    with open(annFile, 'w') as f:
        json.dump(d, f)
        
    print("Done.")