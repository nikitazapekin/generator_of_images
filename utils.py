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
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                images.append(img)
            except Exception as e:
                print(f"Skipping {filename}: {str(e)}")
                continue

    if not images:
        return None

    return torch.stack(images)


def generate_image(model, dataset_file, output_path, device):
    try:
        dataset = torch.load(dataset_file)
        if len(dataset) == 0:
            print("Error: Dataset is empty")
            return False

        idx = torch.randint(0, len(dataset), (1,)).item()
        input_img = dataset[idx].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)

        output_img = output.squeeze(0).cpu().numpy()
        output_img = np.transpose(output_img, (1, 2, 0)) * 255
        output_img = output_img.astype('uint8')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(output_img).save(output_path)
        return True
    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        return False