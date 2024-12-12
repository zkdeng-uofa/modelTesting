import torch
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from typing import Optional, Tuple, Union
from torch import nn, Tensor
from transformers import set_seed, PreTrainedModel, PretrainedConfig, Trainer, AutoImageProcessor, TrainingArguments
from transformers.utils import logging
from transformers.utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices
from evaluate import load
from datasets import load_dataset
from transformers import ConvNextConfig, ConvNextForImageClassification
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention, BaseModelOutputWithNoAttention, BaseModelOutputWithPoolingAndNoAttention
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, Module, _reduction as _Reduction


# Collate function for DataLoader
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



# Image Preprocessor for data augmentation
class ImagePreprocessor():
    def __init__(self, dataset, image_processor):
        self.normalize = transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std,
        )
        self.size = (
            image_processor.size["height"],
            image_processor.size["width"]
        ) if "height" in image_processor.size else (
            image_processor.size["shortest_edge"],
            image_processor.size["shortest_edge"]
        )
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

class ConvNextConfig(PretrainedConfig):
    model_type = "convnext"

    def __init__(
        self,
        num_channels=3,
        patch_size=4,
        num_stages=4,
        hidden_sizes=None,
        depths=None,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.0,
        image_size=224,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]

class ConvNextLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class ConvNextEmbeddings(nn.Module):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        self.layernorm = ConvNextLayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings
    
class ConvNextLayer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.layernorm = ConvNextLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = getattr(nn.functional, config.hidden_act)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_parameter = (
            nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x
    

class ConvNextStage(nn.Module):
    """ConvNeXT stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """

    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()

        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.Sequential(
                ConvNextLayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            )
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.Sequential(
            *[ConvNextLayer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states

class ConvNextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i]
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.stages):
            hidden_states = layer_module(hidden_states)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
    
class ConvNextModel(PreTrainedModel):
    config_class = ConvNextConfig
    base_model_prefix = "convnext"
    main_input_name = "pixel_values"
    _no_split_modules = ["ConvNextLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ConvNextEmbeddings(config)
        self.encoder = ConvNextEncoder(config)

        # final layernorm layer
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))
        
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
    
class ConvNextForImageClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convnext = ConvNextModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.convnext(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)
        loss = None

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss from scratch.

    Args:
        logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels (batch_size).

    Returns:
        torch.Tensor: Computed scalar loss value.
    """
    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=-1)

    # Select the predicted probabilities corresponding to the target class
    batch_size = logits.shape[0]
    target_probs = probs[range(batch_size), targets]

    # Take the log of the probabilities
    log_probs = -torch.log(target_probs + 1e-9)  # Add small value to avoid log(0)

    # Compute the mean loss
    loss = log_probs.mean()
    return loss

def seesaw_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 0.8,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Implements the Seesaw Loss function.

    Args:
        logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels (batch_size).
        alpha (float): Scaling factor for the positive sample term.
        beta (float): Scaling factor for the negative sample term.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        torch.Tensor: Computed scalar loss value.
    """
    num_classes = logits.size(-1)
    batch_size = logits.size(0)

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Create one-hot encoded target tensor
    target_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

    # Positive and negative logits
    pos_logits = logits * target_one_hot
    neg_logits = logits * (1 - target_one_hot)

    # Positive term
    pos_probs = probs * target_one_hot
    pos_loss = -alpha * torch.log(pos_probs + 1e-9) * target_one_hot

    # Negative term
    neg_probs = probs * (1 - target_one_hot)
    neg_factor = torch.pow(1 - neg_probs, beta)
    neg_loss = -neg_factor * torch.log(1 - probs + 1e-9) * (1 - target_one_hot)

    # Total loss
    loss = pos_loss + neg_loss
    if reduction == "mean":
        return loss.sum() / batch_size
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Unpack inputs dictionary and send to device
        pixel_values = inputs["pixel_values"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device)

        # Forward pass through the model
        outputs = model(pixel_values)

        # Compute loss
        logits = outputs.logits
        loss = nn.functional.cross_entropy(logits, labels)
        #loss = cross_entropy(logits, labels)
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
            loss = nn.functional.cross_entropy(logits, labels)
            #loss = cross_entropy(logits, labels)
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
    pretrained_weights_path = "pytorch_model.bin"
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
        eval_strategy="epoch",
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

    trainer.train()

if __name__ == "__main__":
    main()
