import torch
from transformers import Pipeline
from PIL import Image
from typing import Union
from copy import deepcopy
import matplotlib.pyplot as plt
import io

class VLEForVQAPipeline(Pipeline):

    def __init__(self, vle_processor, *args, **kwargs):        
        self.vle_processor = vle_processor
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, top_k=None, **kwargs):
        preprocess_params, forward_params, postprocess_params = {}, {}, {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return preprocess_params, forward_params, postprocess_params

    def __call__(self, image: Union["Image.Image", str], question: str = None, **kwargs):

        if isinstance(image, (Image.Image, str)) and isinstance(question, str):
            inputs = {"image": image, "question": question}
        else:
            """
            Supports the following format
            - {"image": image, "question": question}
            - [{"image": image, "question": question}]
            - Generator and datasets
            """
            inputs = image
        results = super().__call__(inputs, **kwargs)
        return results

    def preprocess(self, inputs):
        model_inputs = self.vle_processor(text=inputs['question'], images=inputs['image'], return_tensors="pt",padding=True)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=1):
        if top_k > self.model.num_vqa_labels:
            top_k = self.model.num_vqa_labels
        probs = torch.softmax(model_outputs['logits'], dim=-1)
        probs, preds = torch.sort(probs, descending=True)
        probs = probs[:,:top_k].tolist()[0]
        preds = preds[:,:top_k].tolist()[0]

        return [{"score": score, "answer": self.model.config.id2label[pred]} for score, pred in zip(probs, preds)]



class VLEForPBCPipeline(Pipeline):
    def __init__(self, vle_processor, *args, **kwargs):        
        self.vle_processor = vle_processor
        self.id2label = {0:"False",1:"True"}
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params, forward_params, postprocess_params = {}, {}, {}
        return preprocess_params, forward_params, postprocess_params

    def __call__(self, image: Union["Image.Image", str], text: str = None, **kwargs):
        if isinstance(image, (Image.Image, str)) and isinstance(text, str):
            inputs = {"image": image, "text": text}
        else:
            """
            Supports the following format
            - {"image": image, "text": text}
            - [{"image": image, "text": text}]
            - Generator and datasets
            """
            inputs = image
        results = super().__call__(inputs, **kwargs)
        return results

    def preprocess(self, inputs):
        model_inputs = self.vle_processor(text=inputs['text'], images=inputs['image'], return_tensors="pt",padding=True)
        return model_inputs, inputs['image']

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs[0])
        return model_outputs, model_inputs[1]

    def postprocess(self, model_outputs):
        probs = torch.softmax(model_outputs[0]['logits'], dim=-1)
        probs = probs.tolist()[0]
        new_image = self.paint_in_image(model_outputs[0]['logits'], model_outputs[1])
        return {"score": probs, "image": new_image}
    
    def paint_in_image(self, logits, raw_image):
        image_back = deepcopy(raw_image)
        raw_image_size = image_back.size
        resized_image_size = self.model.config.vision_config.image_size
        patch_size = self.model.config.vision_config.patch_size
        probs = torch.softmax(logits.detach()[0,:,1].to('cpu'),dim=-1).numpy().reshape(-1, resized_image_size//patch_size)

        plt.close('all')
        plt.axis('off')
        plt.imshow(probs, cmap='gray', interpolation='None', vmin=(probs.max()-probs.min())*2/5+probs.min(),alpha=0.7)
        plt.xticks([])
        plt.yticks([])
        buf = io.BytesIO()
        plt.savefig(buf, dpi=100, transparent=True, bbox_inches='tight', pad_inches=0)
        image_front = Image.open(buf)

        def filter_image_front(img: Image.Image):
            width, height = img.width, img.height
            for x in range(width):
                for y in range(height):
                    r,g,b,a = img.getpixel((x,y))
                    a = int (a * (1-r/255))
                    img.putpixel((x,y), (r,g,b,a))
            return img
        
        image_front = filter_image_front(image_front).resize(raw_image_size)
        image_back.paste(image_front, (0,0), image_front)
        mixed_image = image_back.resize(raw_image_size)
        buf.close()

        return mixed_image



class VLEForITMPipeline(Pipeline):
    def __init__(self, vle_processor, *args, **kwargs):        
        self.vle_processor = vle_processor
        self.id2label = {0:"False",1:"True"}
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params, forward_params, postprocess_params = {}, {}, {}
        return preprocess_params, forward_params, postprocess_params

    def __call__(self, image: Union["Image.Image", str], text: str = None, **kwargs):
        if isinstance(image, (Image.Image, str)) and isinstance(text, str):
            inputs = {"image": image, "text": text}
        else:
            """
            Supports the following format
            - {"image": image, "text": text}
            - [{"image": image, "text": text}]
            - Generator and datasets
            """
            inputs = image
        results = super().__call__(inputs, **kwargs)
        return results

    def preprocess(self, inputs):
        model_inputs = self.vle_processor(text=inputs['text'], images=inputs['image'], return_tensors="pt",padding=True)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs):
        probs = torch.softmax(model_outputs['logits'], dim=-1)
        preds = torch.argmax(probs, dim=-1)
        probs = probs.tolist()[0]
        preds = self.id2label[preds.tolist()[0]]

        return {"score": probs, "match": preds}