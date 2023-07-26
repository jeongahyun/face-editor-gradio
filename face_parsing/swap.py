import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

from .model import BiSeNet


def init_parser(pth_path, device):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    net.load_state_dict(torch.load(pth_path))
    net.eval()
    return net


def image_to_parsing(img, net, device):
    img = cv2.resize(img, (512, 512))
    img = img[:,:,::-1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = transform(img.copy())
    img = torch.unsqueeze(img, 0)

    with torch.no_grad():
        img = img.to(device)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing


def get_mask(parsing, classes):
    res = parsing == classes[0]
    for val in classes[1:]:
        res += parsing == val
    return res


def mask_for_inpaint(source, net, classes, device):
    face_classes = np.unique(classes)
    parsing = image_to_parsing(source, net, device)
    intersection = np.intersect1d(np.unique(parsing), face_classes)
    print('found classes are',np.unique(parsing),'and you want class',face_classes)
    print('then i\'ll give you class', intersection)
    mask = get_mask(parsing, face_classes)
    mask = np.repeat(np.expand_dims(mask, axis=2), 3, 2)
    return cv2.resize(mask.astype(np.uint8)*255, (source.shape[1], source.shape[0]))


def prompt_to_class_by_rule(prompt):
    letter = ['background','skin', 'right-brow', 'left-brow',]
    result = []
    for i, l in enumerate(letter):
        if l in prompt:
            result.append(i)
    return result
