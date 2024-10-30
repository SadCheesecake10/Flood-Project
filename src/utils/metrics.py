import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt

map_metric = MeanAveragePrecision(iou_type="bbox")

def mask_to_boxes(mask):
    boxes = []
    labels = []
    scores = []
    num_objects = mask.max().item()  # Assume that the mask has contiguous labels
    for i in range(1, num_objects + 1):
        pos = torch.where(i == mask)
        if pos[0].numel() > 0:  # Check if the object is has at least one pixel
            xmin = pos[1].min().item()
            xmax = pos[1].max().item()
            ymin = pos[0].min().item()
            ymax = pos[0].max().item()
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Flood label
            scores.append(0.9)  # Confidence score
    return boxes, labels, scores

def evaluate_model(model, dataloader, device):
    model.eval()
    map_metric.reset()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            preds = torch.argmax(preds, dim=1)

            for pred, mask in zip(preds, masks):
                pred_boxes, pred_labels, pred_scores = mask_to_boxes(pred)
                mask_boxes, mask_labels, _ = mask_to_boxes(mask)

                pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32, device=device)
                pred_labels = torch.tensor(pred_labels, dtype=torch.int64, device=device)
                pred_scores = torch.tensor(pred_scores, dtype=torch.float32, device=device)
                mask_boxes = torch.tensor(mask_boxes, dtype=torch.float32, device=device)
                mask_labels = torch.tensor(mask_labels, dtype=torch.int64, device=device)

                predictions = [{"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores}]
                targets = [{"boxes": mask_boxes, "labels": mask_labels}]

                map_metric.update(predictions, targets)

    map_result = map_metric.compute()
    return map_result["map_50"], map_result["map"]
