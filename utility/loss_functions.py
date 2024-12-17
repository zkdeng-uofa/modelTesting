import torch

class LossFunctions():
    def __init__(self):
        pass

    def loss_function(self, loss_name):
        if loss_name == "cross_entropy":
            return cross_entropy
        elif loss_name == "seesaw":
            return seesaw_loss
        else:
            raise ValueError(f"Invalid loss function: {loss_name}")

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
