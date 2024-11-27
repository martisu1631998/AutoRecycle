import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models
from torchvision.models import resnet18, EfficientNet_B0_Weights

# Load the model
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
# model = models.mobilenet_v2()
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 5)  # Adjusted for 5 classes
)
model.load_state_dict(torch.load(r'Best Models/finetuned_v2.pth', map_location=torch.device('cpu')))

model.eval()

##### QUANTIZATION ######
# model_int8 = torch.quantization.quantize_dynamic(
#     model,  # Model to quantize
#     {torch.nn.Linear},  # Layers to quantize
#     dtype=torch.qint8  # Quantization data type
# )
# print("Model quantized successfully!")

#### PRUNING #########
# for module in model.modules():
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name="weight", amount=0.5)  # Prune 50% of weights

# # Remove pruned parameters to finalize the model
# for module in model.modules():
#     if isinstance(module, torch.nn.Conv2d):
#         prune.remove(module, "weight")


#### QUANTIZATION + PRUNING ######
# # Prune the model
# for module in model.modules():
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name="weight", amount=0.5)

# # **Ensure pruning masks are removed** to resolve deepcopy issue
# for module in model.modules():
#     if isinstance(module, torch.nn.Conv2d):
#         prune.remove(module, "weight")

# # Quantize the pruned model
# model_int8 = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )

#### FP16 Precision #####

model.half()
for layer in model.modules():
    if isinstance(layer, nn.BatchNorm2d):
        layer.float()  # Keep batch norm in FP32 for stability


print("Model reduced successfully!")
# Save pruned model to disk
torch.save(model.state_dict(), "Reduced Models/half_finetuned_v2.pth")

# Compare file sizes
original_size = os.path.getsize("Best Models/finetuned_v2.pth")
pruned_size = os.path.getsize("Reduced Models/half_finetuned_v2.pth")

print(f"Original Model Size: {original_size / 1e6:.2f} MB")
print(f"Pruned Model Size: {pruned_size / 1e6:.2f} MB")