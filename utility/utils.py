import torch
import argparse
import json
import numpy as np
import torchvision.transforms as transforms
from evaluate import load
from dataclasses import dataclass, field
from transformers import HfArgumentParser


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    #return pixel_values, labels  # Return as a tuple (image_tensor, label_tensor)
    return {"pixel_values": pixel_values, "labels": labels} 

# Compute metrics for evaluation
def compute_metrics(eval_pred):
    metric_accuracy = load("accuracy")
    metric_f1 = load("f1")
    outputs, labels = eval_pred
    predictions = np.argmax(outputs, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    return {"accuracy": accuracy, "f1": f1}

# Split dataset into train, validation, and test
def split_to_train_val_test(dataset):
    split1 = dataset["train"].train_test_split(test_size=0.2)
    train_ds = split1["train"]
    split2 = split1["test"].train_test_split(test_size=0.5)
    val_ds = split2["train"]
    test_ds = split2["test"]
    return train_ds, val_ds, test_ds

def parse_HF_args():
    """
    Parse hugging face arguments from a JSON file
    """
    # **Added argparse to handle the JSON file path as a command line argument**
    parser = argparse.ArgumentParser(description="Run Hugging Face model with JSON config")
    parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
    args = parser.parse_args()

    # **Load the JSON file specified by the command line argument**
    with open(args.config, 'r') as f:
        json_args = json.load(f)
    
    hf_parser = HfArgumentParser(ScriptTrainingArguments)
    script_args = hf_parser.parse_dict(json_args)
    return script_args[0]  # **Returns the parsed arguments**

@dataclass
class ScriptTrainingArguments:
    """
    Arguments pertaining to this script
    """
    dataset: str = field(
        default=None,
        metadata={"help": "Name of dataset from HG hub"}
    )
    model: str = field(
        default=None,
        metadata={"help": "Name of model from HG hub"}
    )
    learning_rate: float = field(  # **Added learning_rate to the dataclass**
        default=5e-5,
        metadata={"help": "Learning rate for training"}
    )
    num_train_epochs: int = field(  # **Added num_train_epochs to the dataclass**
        default=5,
        metadata={"help": "Number of training epochs"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of training epochs"}
    )
    num_labels: int = field(
        default=5,
        metadata={"help": "Number of training labels"}
    )

# Image Preprocessor for data augmentation
class ImagePreprocessor():
    def __init__(self, dataset, image_processor):
        self.normalize = transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std,
        )

        if "height" in image_processor.size:
            self.size = (
                image_processor.size["height"],
                image_processor.size["width"],
            )
            self.crop_size = self.size
            self.max_size = None
        elif "shortest_edge" in image_processor.size:
            self.size = image_processor.size["shortest_edge"]
            self.crop_size = (self.size, self.size)
            self.max_size = image_processor.size.get("longest_edge")

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                self.normalize
            ]
        )
        self.test_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def preprocess_train(self, image_batch):
        image_batch["pixel_values"] = [
            self.train_transforms(image.convert("RGB")) for image in image_batch["image"]
        ]
        return image_batch
    
    def preprocess_val(self, image_batch):
        image_batch["pixel_values"] = [
            self.val_transforms(image.convert("RGB")) for image in image_batch["image"]
        ]
        return image_batch
    
    def preprocess_test(self, image_batch):
        image_batch["pixel_values"] = [
            self.test_transform(image.convert("RGB")) for image in image_batch["image"]
    ]
        return image_batch