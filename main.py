from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2
import torch.nn as nn

TEST_PATH = "test.jpg"
MODEL_PATH = "model.pth"
DEVICE = "cpu" #cuda/cpu/mps

def test(path):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    img = Image.open(path).convert("RGB")

    transform = v2.Compose([
        v2.Resize((448, 448)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(img_tensor)

    classes = ["digital", "film"]
    pred_class = output.argmax(dim=1).item()
    return classes[pred_class]

if __name__ == "__main__":
    print(test(TEST_PATH))