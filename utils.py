import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def prepare_dataset(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    images = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)

    if not images:
        return None

    dataset = torch.stack(images)
    return dataset


def generate_image(model, dataset_file, output_path, device):
    dataset = torch.load(dataset_file)
    if len(dataset) == 0:
        return False

    # Get a random image from dataset
    idx = torch.randint(0, len(dataset), (1,)).item()
    input_img = dataset[idx].unsqueeze(0).to(device)

    # Generate output
    with torch.no_grad():
        output = model(input_img)

    # Save generated image
    output_img = output.squeeze(0).cpu().numpy()
    output_img = np.transpose(output_img, (1, 2, 0)) * 255
    output_img = output_img.astype('uint8')
    img = Image.fromarray(output_img)
    img.save(output_path)

    return True