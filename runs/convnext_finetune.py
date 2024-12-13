import torch
import os
import torch.nn as nn
from datasets import load_dataset
from transformers import set_seed, TrainingArguments, Trainer, AutoImageProcessor
from transformers.utils import logging

from utility.utils import collate_fn, compute_metrics, split_to_train_val_test, ImagePreprocessor
from models.convnext import ConvNextConfig, ConvNextForImageClassification
from utility.loss_functions import cross_entropy, seesaw_loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Unpack inputs dictionary and send to device
        pixel_values = inputs["pixel_values"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device)

        # Forward pass through the model
        outputs = model(pixel_values)

        # Compute loss
        logits = outputs.logits
        loss = cross_entropy(logits, labels)
        #loss = seesaw_loss(logits, labels)
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
            loss = cross_entropy(logits, labels)
            #loss = seesaw_loss(logits, labels)
        return (loss, logits, labels)


# Main function
def main():
    set_seed(42)
    logger = logging.get_logger(__name__)

    # Load dataset
    dataset = load_dataset("zkdeng/spiderTraining5-100")
    image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")

    # Preprocessing
    image_preprocessor = ImagePreprocessor(dataset, image_processor)
    train_ds, val_ds, test_ds = split_to_train_val_test(dataset)
    train_ds.set_transform(image_preprocessor.preprocess_train)
    val_ds.set_transform(image_preprocessor.preprocess_val)

    # Pretrained weights
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_weights_path = os.path.join(current_dir, '..', 'weights', 'pytorch_model.bin')

    pretrained_weights = torch.load(pretrained_weights_path, map_location="cpu")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(device)
    config = ConvNextConfig(num_labels=5, depths=[3, 3, 9, 3])
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

    training_args = TrainingArguments(
        output_dir="convnext-checkpoints",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,  # AutoImageProcessor works as a tokenizer here for image tasks
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    train_results = trainer.train()
    #trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    test_ds.set_transform(image_preprocessor.preprocess_test)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
