#Finetuning the DETR model using the CPPE-5 dataset to improve its performance for a specific taske of helath mask detection

#This example has been adapted to run locally.

#Source: https://huggingface.co/docs/transformers/en/tasks/object_detection

# Transformers installation
# pip install datasets evaluate timm albumentations
# pip install git+https://github.com/huggingface/transformers.git
# pip install git+https://github.com/huggingface/accelerate.git


#The [CPPE-5 dataset](https://huggingface.co/datasets/cppe-5) contains images with
#annotations identifying medical personal protective equipment (PPE) in the context of the COVID-19 pandemic.

from datasets import load_dataset
from PIL import Image, ImageDraw
import numpy as np
import os

#start config
train_model = True
use_only_2_images_for_training = True #Warning if TRUE not all images are used for Training. This is for tesing only.
test_image = True
evaluate_new_funetuned_model = False
#end config

#fune tuned model folder
model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"fine_tuned_model")

cppe5 = load_dataset("cppe-5")

categories = cppe5["train"].features["objects"].feature["category"].names
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

#Remove bad data
remove_idx = [590, 821, 822, 875, 876, 878, 879]
keep = [i for i in range(len(cppe5["train"])) if i not in remove_idx]
if use_only_2_images_for_training:
    keep = keep[0:2]
cppe5["train"] = cppe5["train"].select(keep)

#Instantiate the image processor from the same checkpoint as the model you want to finetune.
from transformers import AutoImageProcessor
checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

#Before passing the images to the image_processor, apply two preprocessing transformations to the dataset:
#Augmenting images
#Reformatting annotations to meet DETR expectations

import albumentations
import numpy as np
import torch

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

#The image_processor expects the annotations to be in the following format: {'image_id': int, 'annotations': List[Dict]}
#add a function to reformat annotations for a single example:
def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

# combine the image and annotation transformations to use on a batch of examples 
# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")
    
#Apply this preprocessing function to the entire dataset using Datasets with_transform method
cppe5["train"] = cppe5["train"].with_transform(transform_aug_ann)

#You have successfully augmented the individual images and prepared their annotations. However, preprocessing isn't complete yet. 
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

#Training involves the following steps:

# - Load the model with AutoModelForObjectDetection using the same checkpoint as in the preprocessing.
# - Define your training hyperparameters in TrainingArguments.
# - Pass the training arguments to Trainer along with the model, dataset, image processor, and data collator.
# - Call train() to finetune your model.

from transformers import AutoModelForObjectDetection
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="detr-resnet-50_finetuned_cppe5",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

#Finally perform the training

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=cppe5["train"],
    tokenizer=image_processor,
)

if train_model:
    
    print ("Starting training (fine-tuning) ...")
    trainer.train()

    #save model to disk
    print("Saving model ...")
    trainer.save_model(model_folder)
    
#Test on image==============================================================================================

if test_image:
    from transformers import pipeline
    import requests
    
    print("Load model from disk ...")
    fine_tuned_model = AutoModelForObjectDetection.from_pretrained(model_folder)
    image_processor_fine_tuned = AutoImageProcessor.from_pretrained(model_folder)
    
    url = "https://image.cnbcfm.com/api/v1/image/106467352-1585602933667virus-medical-flu-mask-health-protection-woman-young-outdoor-sick-pollution-protective-danger-face_t20_o07dbe.jpg?v=1585602987&w=929&h=523&vtcrop=y"
    image = Image.open(requests.get(url, stream=True).raw)
    
    
    obj_detector = pipeline("object-detection", model=fine_tuned_model, image_processor = image_processor_fine_tuned)
    #obj_detector = pipeline("object-detection", model="MariaK/detr-resnet-50_finetuned_cppe5")
    obj_detector(image)

#Evaluation after training =================================================================================
#Object detection models are commonly evaluated with a set of COCO-style metrics.

if evaluate_new_funetuned_model:
    #Preparation 1 for evaluation
    import json
    # format annotations the same as for training, no need for data augmentation
    def val_formatted_anns(image_id, objects):
        annotations = []
        for i in range(0, len(objects["id"])):
            new_ann = {
                "id": objects["id"][i],
                "category_id": objects["category"][i],
                "iscrowd": 0,
                "image_id": image_id,
                "area": objects["area"][i],
                "bbox": objects["bbox"][i],
            }
            annotations.append(new_ann)
    
        return annotations
    
    
    # Save images and annotations into the files torchvision.datasets.CocoDetection expects
    def save_cppe5_annotation_file_images(cppe5):
        output_json = {}
        path_output_cppe5 = f"{os.getcwd()}/cppe5/"
    
        if not os.path.exists(path_output_cppe5):
            os.makedirs(path_output_cppe5)
    
        path_anno = os.path.join(path_output_cppe5, "cppe5_ann.json")
        categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
        output_json["images"] = []
        output_json["annotations"] = []
        for example in cppe5:
            ann = val_formatted_anns(example["image_id"], example["objects"])
            output_json["images"].append(
                {
                    "id": example["image_id"],
                    "width": example["image"].width,
                    "height": example["image"].height,
                    "file_name": f"{example['image_id']}.png",
                }
            )
            output_json["annotations"].extend(ann)
        output_json["categories"] = categories_json
    
        with open(path_anno, "w") as file:
            json.dump(output_json, file, ensure_ascii=False, indent=4)
    
        for im, img_id in zip(cppe5["image"], cppe5["image_id"]):
            path_img = os.path.join(path_output_cppe5, f"{img_id}.png")
            im.save(path_img)
    
        return path_output_cppe5, path_anno
    
    #Preparation 2 for evaluation
    import torchvision
    class CocoDetection(torchvision.datasets.CocoDetection):
        def __init__(self, img_folder, feature_extractor, ann_file):
            super().__init__(img_folder, ann_file)
            self.feature_extractor = feature_extractor
    
        def __getitem__(self, idx):
            # read in PIL image and target in COCO format
            img, target = super(CocoDetection, self).__getitem__(idx)
    
            # preprocess image and target: converting target to DETR format,
            # resizing + normalization of both image and target)
            image_id = self.ids[idx]
            target = {"image_id": image_id, "annotations": target}
            encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
            target = encoding["labels"][0]  # remove batch dimension
    
            return {"pixel_values": pixel_values, "labels": target}
    
    
    im_processor = AutoModelForObjectDetection.from_pretrained(model_folder) #AutoImageProcessor.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")
    
    path_output_cppe5, path_anno = save_cppe5_annotation_file_images(cppe5["test"])
    test_ds_coco_format = CocoDetection(path_output_cppe5, im_processor, path_anno)
    
    #3 Finally run the evaluation
    import evaluate
    from tqdm import tqdm
    
    model = AutoModelForObjectDetection.from_pretrained(model_folder) #AutoModelForObjectDetection.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")
    module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    val_dataloader = torch.utils.data.DataLoader(
        test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]
    
            labels = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  # these are in DETR format, resized + normalized
    
            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = im_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api
    
            module.add(prediction=results, reference=labels)
            del batch
    
    results = module.compute()
    print(results)