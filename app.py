import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from io import BytesIO
import requests
from face_parsing import init_parser, mask_for_inpaint, prompt_to_class_by_rule
import cv2
import numpy as np

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def face_parse(face, classes, outpath, seg_net, device):
    f = cv2.imread(face)
    c = classes
    face_mask = cv2.resize(mask_for_inpaint(f, seg_net, c, device),f.shape[:2])
    kernel = np.ones((3,3), np.uint8)
    mask_dilation = cv2.dilate(face_mask, kernel, iterations=7)
    cv2.imwrite(outpath, mask_dilation)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sd_model = "runwayml/stable-diffusion-v1-5"
inpaint_model = "runwayml/stable-diffusion-inpainting"
parse_model = "face_parsing/model/face_segmentation.pth"
mask_path = "data/mask.png"
img_path = "data/input.png"

pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model, torch_dtype=torch.float16)
pipe_img2img = pipe_img2img.to(device)

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(inpaint_model, torch_dtype=torch.float16)
pipe_inpaint = pipe_inpaint.to(device)

def inference(img, eyes, lips):
    img.save(img_path)
    seg_net = init_parser(parse_model, device=device)
    out = img
    print(type(eyes))
    print(len(eyes))
    print(eyes == "")
    if eyes != "default":
        face_parse(img_path, [4,5], mask_path, seg_net, device)
        mask_img = Image.open(mask_path)
        out = pipe_inpaint(prompt=eyes, image=img, mask_image=mask_img, guidance_scale=5).images[0]
        img = out
    if lips != "default":
        face_parse(img_path, [11,12,13], mask_path, seg_net, device)
        mask_img = Image.open(mask_path)
        out = pipe_inpaint(prompt=lips, image=img, mask_image=mask_img, guidance_scale=5).images[0]
    return out

title = "Face Editor"
description = "stable diffusion을 사용한 인물 편집 툴입니다."
radio_eyes = gr.Radio(
    ["default", "slanted eyes", "droopy eyes", "sharp eyes"], label="눈 모양"
)
radio_lips = gr.Radio(
    ["default", "plumpy lips", "small slim lips", "smiling lips", "angry lips"], label="입술 모양"
)

demo = gr.Interface(
    fn=inference,
    inputs=[gr.inputs.Image(type="pil"), radio_eyes, radio_lips],
    outputs=gr.outputs.Image(type="pil"),
    title=title,
    description=description
)

demo.launch(server_name='0.0.0.0')