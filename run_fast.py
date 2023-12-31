#!/usr/bin/env python3
import argparse
import torch
import os
import open_clip
from PIL import Image
from clip_interrogator import Config, Interrogator

import base64

base_str = "U2xlZWVlcHlaaG915pW05ZCI5YWx5Lqr"
utf8_str = base64.b64decode(base_str).decode("utf-8")

class BatchWriter:
    def __init__(self, folder):
        self.folder = folder
        self.file = None
        
    def add(self, file, prompt):
        txt_file = os.path.splitext(file)[0] + ".txt"
        with open(os.path.join(self.folder, txt_file), 'w', encoding='utf-8') as f:
            f.write(prompt)

    def close(self):
        if self.file is not None:
            self.file.close()
            
def batch_process(folder):
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist")
            return
        if not os.path.isdir(folder):
            print("{folder} is not a directory")
            return

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print("Folder has no images")
            return

        try:
            ci.config.quiet = True
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

            writer = BatchWriter(folder)
            for idx, file in enumerate(files):
                try:
                    image = Image.open(os.path.join(folder, file)).convert('RGB')
                    prompt = ci.interrogate_fast(image,caption=captions[idx])
                    writer.add(file, prompt)
                except OSError as e:
                    print(f" {e}, continuing")
            writer.close()
            ci.config.quiet = False
            print("Finished")
            
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            print("Out of VRAM!")
            return
        except RuntimeError as e:
            print(e)
            return

parser = argparse.ArgumentParser()
parser.add_argument("--lowvram", action='store_true', help="Optimize settings for low VRAM")
parser.add_argument("--i_path", type=str, default=None, help="images path")
args = parser.parse_args()

print(utf8_str)
if not torch.cuda.is_available():
    print("CUDA is not available, using CPU. Warning: this will be very slow!")

config = Config(cache_path="cache")

if args.lowvram:
    config.apply_low_vram_defaults()
ci = Interrogator(config)

batch_process(args.i_path)