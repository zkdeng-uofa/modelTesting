import torch
import os
import wandb
import torch.nn as nn
from datasets import load_dataset
from transformers import set_seed, TrainingArguments, Trainer, AutoImageProcessor
from transformers.utils import logging

from models.convnext import ConvNextConfig, ConvNextForImageClassification
from utility.loss_functions import seesaw_loss, LossFunctions
from utility.utils import (
    collate_fn, 
    compute_metrics, 
    parse_HF_args, 
    preprocess_hf_dataset,
    preprocess_kg_dataset,
    preprocess_hf_ros_dataset,
)

class CustomTrainer(Trainer):
    def __init__(self, loss_fxn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lossClass = LossFunctions()
        self.loss_fxn = loss_fxn

    def compute_loss(self, model, inputs, return_outputs=False):
        # Unpack inputs dictionary and send to device
        pixel_values = inputs["pixel_values"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device)

        # Forward pass through the model
        outputs = model(pixel_values)

        # Compute loss
        logits = outputs.logits
        loss_fxn = self.lossClass.loss_function(self.loss_fxn)
        loss = loss_fxn(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override the prediction_step to properly handle evaluation (unpacking dict).
        """
        pixel_values = inputs["pixel_values"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device) if "labels" in inputs else None

        with torch.no_grad():
            outputs = model(pixel_values)

        logits = outputs.logits
        loss = None
        if labels is not None:
            #loss = seesaw_loss(logits, labels)
            loss_fxn = self.lossClass.loss_function(self.loss_fxn)
            loss = loss_fxn(logits, labels)
        return (loss, logits, labels)


# Main function
def main(script_args):
    set_seed(42)

    if script_args.wandb == "True":
        wandb.login(key="e68d14a1a7b3aed71e0455589cde53c783018f5a")
        wandb.init(
            project="convnext",
            name=script_args.output_dir,
        )

        wandb.config.update({
            "model_checkpoint": script_args.output_dir,
            "batch_size": script_args.batch_size,
            "learning_rate": script_args.learning_rate,  # **Updated to use value from JSON**
            "num_train_epochs": script_args.num_train_epochs,  # **Updated to use value from JSON**
        })
    #Load dataset

    image_processor = AutoImageProcessor.from_pretrained(script_args.model)


    train_ds, val_ds, test_ds = preprocess_hf_ros_dataset(script_args.dataset, script_args.model)
    # if script_args.dataset_host == "huggingface":
    #     train_ds, val_ds, test_ds = preprocess_hf_dataset(script_args.dataset, script_args.model)
    # elif script_args.dataset_host == "kaggle":
    #     train_ds, val_ds, test_ds = preprocess_kg_dataset(
    #         script_args.dataset, 
    #         script_args.local_dataset_name, 
    #         script_args.model
    #     )
    # Pretrained weights

    print(train_ds)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_weights_path = os.path.join(current_dir, '..', 'weights', 'pytorch_model.bin')

    pretrained_weights = torch.load(pretrained_weights_path, map_location="cpu")

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    config = ConvNextConfig(num_labels=script_args.num_labels, depths=[3, 3, 9, 3])
    model = ConvNextForImageClassification(config)
    model.to(device)

    # Load pretrained weights
    filtered_weights = {k: v for k, v in pretrained_weights.items() if "classifier" not in k}
    missing_keys, unexpected_keys = model.load_state_dict(filtered_weights, strict=False)

    # Check for any missing or unexpected keys
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    print("Pretrained weights loaded successfully!")

    if script_args.wandb == "True":
        training_args = TrainingArguments(
            output_dir=f"checkpoints/{script_args.output_dir}",
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=script_args.learning_rate,
            per_device_train_batch_size=script_args.batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=script_args.batch_size,
            num_train_epochs=script_args.num_train_epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=(script_args.push_to_hub=="True"),
            report_to="wandb",
        )
    else:
        training_args = TrainingArguments(
            output_dir=f"checkpoints/{script_args.output_dir}",
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=script_args.learning_rate,
            per_device_train_batch_size=script_args.batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=script_args.batch_size,
            num_train_epochs=script_args.num_train_epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=(script_args.push_to_hub=="True"),
        )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,  # AutoImageProcessor works as a tokenizer here for image tasks
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        loss_fxn=script_args.loss_function
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        loss_fxn=script_args.loss_function
    )

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    set_seed(42)
    logger = logging.get_logger(__name__)
    args = parse_HF_args()  # **Updated to use JSON-based arguments**
    main(args)
