import os
import torch
import numpy as np
from PIL import Image
from skimage import io, transform
from skimage.filters import gaussian
from torch.autograd import Variable
from torchvision import transforms
from model import U2NET
import cv2


# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def apply_u2net_portrait(image, sigma, alpha, model_path="u2net_portrait.pth"):
    # Load the model
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()

    # Preprocess the image
    # image = Image.open(image_path).convert("RGB")
    transform_to_tensor = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    # print(type(image))
    # image = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)
    input_image = transform_to_tensor(image).unsqueeze(0)

    if torch.cuda.is_available():
        model.cuda()
        input_image = input_image.cuda()

    # Inference
    with torch.no_grad():
        d1, _, _, _, _, _, _ = model(input_image)
        pred = 1.0 - d1[:, 0, :, :]
        pred = normPRED(pred)

    # Post-process and create the final image
    pred_np = pred.squeeze().cpu().data.numpy()
    pd_resized = transform.resize(pred_np, image.size[::-1], order=2)
    pd_resized = pd_resized / (np.amax(pd_resized) + 1e-8) * 255
    pd_resized = pd_resized[:, :, np.newaxis]

    # Gaussian blur and fusion with the original image
    original_image = np.array(image)
    blurred_image = gaussian(original_image, sigma=sigma, preserve_range=True)
    composite_image = blurred_image * alpha + pd_resized * (1 - alpha)

    # Convert back to PIL image
    final_image = Image.fromarray(composite_image.astype(np.uint8))
    return final_image
