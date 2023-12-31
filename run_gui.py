#!/usr/bin/env python3
import argparse
import torch
import os
import csv
import open_clip
from PIL import Image
from clip_interrogator import Config, Interrogator, list_caption_models, list_clip_models
import base64

base_str = "U2xlZWVlcHlaaG915pW05ZCI5YWx5Lqr"
utf8_str = base64.b64decode(base_str).decode("utf-8")
try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed, please install it with 'pip install gradio'")
    exit(1)

def get_models():
    return ['/'.join(x) for x in open_clip.list_pretrained()]

def image_analysis(image, clip_model_name):
    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()

    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)

    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}
    
    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks

def image_to_prompt(image, mode, clip_model_name, blip_model_name):
    try:
        if blip_model_name != ci.config.caption_model_name:
            ci.config.caption_model_name = blip_model_name
            ci.load_caption_model()

        if clip_model_name != ci.config.clip_model_name:
            ci.config.clip_model_name = clip_model_name
            ci.load_clip_model()

        image = image.convert('RGB')
        if mode == 'best':
            return ci.interrogate(image)
        elif mode == 'classic':
            return ci.interrogate_classic(image)
        elif mode == 'fast':
            return ci.interrogate_fast(image)
        elif mode == 'negative':
            return ci.interrogate_negative(image)
    except torch.cuda.OutOfMemoryError as e:
        prompt = "Ran out of VRAM"
        print(e)
        return prompt
    
class BatchWriter:
    def __init__(self, folder, mode):
        self.folder = folder
        self.mode = mode
        self.csv, self.file = None, None
        if mode == BATCH_OUTPUT_MODES[1]:
            self.file = open(os.path.join(folder, 'batch.txt'), 'w', encoding='utf-8')
        elif mode == BATCH_OUTPUT_MODES[2]:
            self.file = open(os.path.join(folder, 'batch.csv'), 'w', encoding='utf-8', newline='')
            self.csv = csv.writer(self.file, quoting=csv.QUOTE_MINIMAL)
            self.csv.writerow(['filename', 'prompt'])

    def add(self, file, prompt):
        if self.mode == BATCH_OUTPUT_MODES[0]:
            txt_file = os.path.splitext(file)[0] + ".txt"
            with open(os.path.join(self.folder, txt_file), 'w', encoding='utf-8') as f:
                f.write(prompt)
        elif self.mode == BATCH_OUTPUT_MODES[1]:
            self.file.write(f"{prompt}\n")
        elif self.mode == BATCH_OUTPUT_MODES[2]:
            self.csv.writerow([file, prompt])

    def close(self):
        if self.file is not None:
            self.file.close()
            
def prompt_tab():
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            with gr.Column():
                mode = gr.Radio(['best', 'fast', 'classic', 'negative'], label='Mode', value='best')
                clip_model = gr.Dropdown(list_clip_models(), value=ci.config.clip_model_name, label='CLIP Model')
                blip_model = gr.Dropdown(list_caption_models(), value=ci.config.caption_model_name, label='Caption Model')
        prompt = gr.Textbox(label="Prompt")
    button = gr.Button("Generate prompt")
    button.click(image_to_prompt, inputs=[image, mode, clip_model, blip_model], outputs=prompt)

def analyze_tab():
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            model = gr.Dropdown(list_clip_models(), value='ViT-L-14/openai', label='CLIP Model')
        with gr.Row():
            medium = gr.Label(label="Medium", num_top_classes=5)
            artist = gr.Label(label="Artist", num_top_classes=5)        
            movement = gr.Label(label="Movement", num_top_classes=5)
            trending = gr.Label(label="Trending", num_top_classes=5)
            flavor = gr.Label(label="Flavor", num_top_classes=5)
    button = gr.Button("Analyze")
    button.click(image_analysis, inputs=[image, model], outputs=[medium, artist, movement, trending, flavor])

def batch_tab():
    def batch_process(folder, mode, clip_model_name, blip_model_name, output_mode):
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist")
            return "tan90(目录不存在)"
        if not os.path.isdir(folder):
            print("{folder} is not a directory")
            return "你给了我一个假路径(输入非路径，请检查)"

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print("Folder has no images")
            return "俺的图图呢？(目录内未识别出图片)"

        try:
            if blip_model_name != ci.config.caption_model_name:
                ci.config.caption_model_name = blip_model_name
                ci.load_caption_model()
            if clip_model_name != ci.config.clip_model_name:
                ci.config.clip_model_name = clip_model_name
                ci.load_clip_model()

            captions = []
            for file in files:
                try:
                    image = Image.open(os.path.join(folder, file)).convert('RGB')
                    caption = ci.generate_caption(image)
                except OSError as e:
                    print(f"{e}; continuing")
                    caption = ""
                finally:
                    captions.append(caption)

            writer = BatchWriter(folder, output_mode)
            for idx, file in enumerate(files):
                try:
                    image = Image.open(os.path.join(folder, file)).convert('RGB')
                    if mode == 'best':
                        prompt = ci.interrogate(image,caption=captions[idx])
                    elif mode == 'classic':
                        prompt = ci.interrogate_classic(image,caption=captions[idx])
                    elif mode == 'fast':
                        prompt = ci.interrogate_fast(image,caption=captions[idx])
                    elif mode == 'negative':
                        prompt = ci.interrogate_negative(image,caption=captions[idx])
                    writer.add(file, prompt)
                except OSError as e:
                    print(f" {e}, continuing")
            writer.close()
            return "打标完成"
            
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            print("Out of VRAM!")
            return "显存爆炸"
        except RuntimeError as e:
            print(e)
            return "RuntimeError"

    with gr.Column():
        with gr.Row():
            folder = gr.Text(label="图片文件夹路径", value="", interactive=True)
        with gr.Row():
            mode = gr.Radio(['best', 'fast', 'classic', 'negative'], label='Prompt Mode', value='fast')
            clip_model = gr.Dropdown(list_clip_models(), value=ci.config.clip_model_name, label='CLIP Model')
            blip_model = gr.Dropdown(list_caption_models(), value=ci.config.caption_model_name, label='Caption Model')
            output_mode = gr.Dropdown(BATCH_OUTPUT_MODES, value=BATCH_OUTPUT_MODES[0], label='Output Mode')
        with gr.Row():
            button = gr.Button("Go!", variant='primary')
            state = gr.Textbox(label="目前状态")

    button.click(batch_process, inputs=[folder, mode, clip_model, blip_model, output_mode], outputs=state)

def about_tab():
    gr.Markdown("## 🕵️‍♂️ CLIP 反推 🕵️‍♂️")
    gr.Markdown("## 注意事项")
    gr.Markdown(
        "CLIP 模型:\n"
        "* 要获得 Stable Diffusion 1.x 的最佳提示词，请选择 **ViT-L-14/openai** 模型.\n"
        "* 要获得 Stable Diffusion 2.x 的最佳提示词，请选择 **ViT-H-14/laion2b_s32b_b79k** 模型.\n"
        "* 要获得 Stable Diffusion XL 的最佳提示词，请选择 **ViT-L-14/openai** 或者 **ViT-bigG-14/laion2b_s39b** 模型.\n"
    )
    gr.Markdown("## Github")
    gr.Markdown("如对此项目有任何问题欢迎浏览GitHub上的[CLIP Interrogator on Github](https://github.com/pharmapsychotic/clip-interrogator) ,如果喜欢本插件的话还请点个 Star !")
    gr.Markdown("## 关于整合")
    gr.Markdown("整合包有相关问题，欢迎[在这里](https://space.bilibili.com/360375877) 私信给我反馈")
    gr.Markdown("整合包本是为方便自用，寻思反正做了就发出来了，如果感觉好用[在此](https://www.bilibili.com/video/BV19b4y1G7Hs)蹲个三连关注")
    gr.Markdown("整合自开源项目，感谢所有开源作者")

BATCH_OUTPUT_MODES = [
    '每个图片文件一个txt',
    '单个txt包含所有tag',
    'csv表格',
]

parser = argparse.ArgumentParser()
parser.add_argument("--lowvram", action='store_true', help="Optimize settings for low VRAM")
parser.add_argument('-s', '--share', action='store_true', help='Create a public link')
args = parser.parse_args()

print(utf8_str)
if not torch.cuda.is_available():
    print("CUDA is not available, using CPU. Warning: this will be very slow!")

config = Config(cache_path="cache")

if args.lowvram:
    config.apply_low_vram_defaults()
ci = Interrogator(config)

with gr.Blocks() as ui:
    gr.Markdown("# <center>🕵️‍♂️ CLIP Interrogator 🕵️‍♂️</center>")
    with gr.Tab("反推提示词"):
        prompt_tab()
    with gr.Tab("图片分析"):
        analyze_tab()
    with gr.Tab("为图片集生成Tag"):
        batch_tab()
    with gr.Tab("关于"):
        about_tab()

ui.launch(server_port=5001, show_api=True, debug=True, share=args.share)