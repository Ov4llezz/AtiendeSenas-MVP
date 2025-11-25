import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

ckpt = "MCG-NJU/videomae-base"

processor = VideoMAEImageProcessor.from_pretrained(ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    ckpt,
    num_labels=100,           # WLASL100
    ignore_mismatched_sizes=True,
)

print("Modelo y processor cargados OK.")
print("Num labels:", model.config.num_labels)
print("Device:", next(model.parameters()).device)


