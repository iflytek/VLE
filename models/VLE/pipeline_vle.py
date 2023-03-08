import torch
from transformers import Pipeline, BatchEncoding
from PIL import Image
from typing import Union
from copy import deepcopy
import matplotlib.pyplot as plt
import io
import json
import unicodedata
import os

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


class VLEForVCRQ2APipeline(Pipeline):

    def __init__(self, vle_processor, *args, **kwargs):        
        self.vle_processor = vle_processor
        self.vle_tokenizer = self.vle_processor.tokenizer
        self.GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Payton', 'Skyler', 'Frankie', 'Pat', 'Quinn']
        self.person_name_id = 0
        self.max_text_len = 80
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params, forward_params, postprocess_params = {}, {}, {}
        return preprocess_params, forward_params, postprocess_params

    def __call__(self, vcr_image_root: str, meta_inputs: dict, **kwargs):

        inputs = {"vcr_image_root": vcr_image_root, "meta_inputs": meta_inputs}
        results = super().__call__(inputs, **kwargs)
        return results

    def preprocess(self, inputs):
        model_inputs = self.vcr_q2a_preprocess(inputs["vcr_image_root"], inputs["meta_inputs"])
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=1):
        logits = model_outputs["logits"]
        loss = model_outputs["loss"]
        preds = torch.argmax(logits, dim=-1, keepdim=True)

        return [{"score": score, "pred": pred} for score, pred in zip(logits, preds)]

    def vcr_q2a_preprocess(self, vcr_image_root, data):
        image_fn = data["img_fn"]
        objects = data["objects"]
        metadata_fn = data["metadata_fn"]
        question = data["question"]
        answer_choices = data["answer_choices"]
        rationale_choices = data["rationale_choices"]
        answer_label = data["answer_label"]
        rationale_label = data["rationale_label"]

        question_text = question
        answer_text = answer_choices
        text_tokens, text_ids, obj_tags, text_raw = self.build_text(question_text, answer_text, objects, self.vle_tokenizer)
        encoding = [self.vle_tokenizer(
            ''.join(self.vle_tokenizer.convert_ids_to_tokens(text_ids_)),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        ) for text_ids_ in text_ids]

        obj_tags = [[-1] + tags + [-1] for tags in obj_tags]

        image = Image.open(os.path.join(vcr_image_root, image_fn))
        image_feature = self.vle_processor.image_processor(image)
        width = image.size[0]
        height = image.size[1]
        with open(os.path.join(vcr_image_root, metadata_fn), 'r') as f:
            vcr_image_metadata = json.load(f)
        boxes = vcr_image_metadata['boxes']
        patch_boxes = self.get_patch_box(boxes, width, height, self.model.config.vision_config.patch_size, self.model.config.vision_config.image_size)
        related_box_ids, image_added_token_type_ids, text_token_type_ids = list(zip(*[self.get_related_box_ids_and_token_type_ids(tags) for tags in obj_tags]))
        related_patch_boxes = [[patch_boxes[i] for i in related_box_ids[j]] for j in range(4)]

        processed_data = {
            "image": image_feature,
            "text": (text_raw, encoding),
            "obj_tags": obj_tags,
            "label": answer_label,
            "patch_ids": [[patch_box[1] for patch_box in related_patch_boxes[i]] for i in range(4)],
            "extend_token_type_ids": [(image_added_token_type_ids[i], text_token_type_ids[i]) for i in range(4)],
        }

        model_inputs = {
            "input_ids": [[processed_data["text"][1][i]["input_ids"]] for i in range(4)],
            "attention_mask": [[processed_data["text"][1][i]["attention_mask"]] for i in range(4)],
            "token_type_ids": [[processed_data["text"][1][i]["token_type_ids"]] for i in range(4)],
            "pixel_values": torch.Tensor([processed_data["image"]["pixel_values"][0]]),
        }
        model_inputs = BatchEncoding(model_inputs, tensor_type='pt')
        model_inputs.update({
            "patch_ids": [[processed_data["patch_ids"][i]] for i in range(4)],
            "vcr_labels": [processed_data["label"]],
            "extend_token_type_ids": [list(zip(processed_data["extend_token_type_ids"][i])) for i in range(4)],
            "return_loss": True,
        })
        return model_inputs

    def retokenize_and_convert_to_ids_with_tag(self, raw_tokens, objects_replace_name, tokenizer, non_obj_tag=-1, add_space_b4_first_token=False):
        parsed_tokens = []
        tags = []
        align_ids = []
        raw = []
        align_id = 0
        for idx, mixed_token in enumerate(raw_tokens):
            if isinstance(mixed_token, list):
                tokens = [" " + objects_replace_name[o] for o in mixed_token]
                if idx == 0 and not add_space_b4_first_token:
                    tokens[0] = tokens[0].lstrip()
                retokenized_tokens = tokenizer.tokenize(tokens[0])
                raw.append(tokens[0])
                tags.extend([mixed_token[0] + non_obj_tag + 1 for _ in retokenized_tokens])
                align_ids.extend([align_id for _ in retokenized_tokens])
                align_id += 1
                for token, o in zip(tokens[1:], mixed_token[1:]):
                    retokenized_tokens.append(tokenizer.tokenize(' and')[0])
                    tags.append(non_obj_tag)
                    align_ids.append(align_id)
                    align_id += 1
                    re_tokens = tokenizer.tokenize(token)
                    retokenized_tokens.extend(re_tokens)
                    tags.extend([o + non_obj_tag + 1 for _ in re_tokens])
                    align_ids.extend([align_id for _ in re_tokens])
                    align_id += 1
                    raw.extend([' and', token])
                parsed_tokens.extend(retokenized_tokens)
            else:
                # fully align to original tokens
                if True in [unicodedata.category(str_) == 'Co' for str_ in mixed_token]:
                    continue
                if idx != 0 or add_space_b4_first_token:
                    mixed_token = " " + mixed_token
                raw.append(mixed_token)
                retokenized_tokens = tokenizer.tokenize(mixed_token)
                parsed_tokens.extend(retokenized_tokens)
                align_ids.extend([align_id for _ in retokenized_tokens])
                tags.extend([non_obj_tag for _ in retokenized_tokens])
                align_id += 1
        ids = tokenizer.convert_tokens_to_ids(parsed_tokens)
        ids_with_tag = list(zip(parsed_tokens, ids, tags, align_ids))
        
        return ids_with_tag, raw

    def build_text(self, question_text, answer_text, objects, tokenizer):
        objects_replace_name = []
        for o in objects:
            if o == 'person':
                objects_replace_name.append(self.GENDER_NEUTRAL_NAMES[self.person_name_id])
                self.person_name_id = (self.person_name_id + 1) % len(self.GENDER_NEUTRAL_NAMES)
            else:
                objects_replace_name.append(o)

        non_obj_tag = -1
        question_text, question_text_raw = self.retokenize_and_convert_to_ids_with_tag(question_text, objects_replace_name, tokenizer, non_obj_tag=non_obj_tag)
        answer_text = [self.retokenize_and_convert_to_ids_with_tag(a_t, objects_replace_name, tokenizer, non_obj_tag=non_obj_tag) for a_t in answer_text]
        answer_text, answer_text_raw = list(zip(*answer_text))
        for a_t, a_t_raw in zip(answer_text, answer_text_raw):
            while len(question_text) + len(a_t) > self.max_text_len - 3:
                if len(question_text) > len(a_t):
                    question_text.pop()
                else:
                    a_t.pop()

        text_tokens = [[q_t[0] for q_t in question_text] + [self.vle_tokenizer.sep_token] + [a_t_t[0] for a_t_t in a_t] for a_t in answer_text]
        text_ids = [[q_t[1] for q_t in question_text] + [self.vle_tokenizer.sep_token_id] + [a_t_t[1] for a_t_t in a_t] for a_t in answer_text]
        obj_tags = [[q_t[2] for q_t in question_text] + [-1] + [a_t_t[2] for a_t_t in a_t] for a_t in answer_text]
        text_raw = [question_text_raw + answer_text_raw_ for answer_text_raw_ in answer_text_raw]
        
        return text_tokens, text_ids, obj_tags, text_raw

    def get_patch_box(self, boxes, width, height, patch_size, image_size):
        patch_count_w = image_size // patch_size
        patch_count_h = image_size // patch_size
        patch_width = width / patch_count_w
        patch_height = height / patch_count_h

        patch_boxes = []
        for box in boxes:
            box = box[:4]
            patch_x1 = int(box[0] // patch_width)
            patch_y1 = int(box[1] // patch_height)
            patch_x2 = int(box[2] // patch_width)
            patch_y2 = int(box[3] // patch_height)

            patch_x1 = patch_x1 if patch_x1 >= 0 else 0
            patch_y1 = patch_y1 if patch_y1 >= 0 else 0
            patch_x2 = patch_x2 + 1 if patch_x2 < patch_count_w else patch_count_w
            patch_y2 = patch_y2 + 1 if patch_y2 < patch_count_h else patch_count_h

            patch_box = [
                patch_x1 * patch_width,
                patch_y1 * patch_height,
                patch_x2 * patch_width,
                patch_y2 * patch_height
            ]

            patch_ids = [patch_count_w * y + x for y in range(patch_y1, patch_y2) for x in range(patch_x1, patch_x2)]
            patch_boxes.append([patch_box, patch_ids])

        return patch_boxes

    def get_related_box_ids_and_token_type_ids(self, obj_tags):
        no_obj_tag = -1
        obj_tags_set = set()
        for tag in obj_tags:
            if tag != no_obj_tag:
                obj_tags_set.add(tag)
        obj_tag_remap = {t: i + 2 for i, t in enumerate(obj_tags_set)}
        text_token_type_ids = [obj_tag_remap[tag] if tag != no_obj_tag else 0 for tag in obj_tags]
        related_box_ids = list(obj_tag_remap.keys())
        image_added_token_type_ids = list(obj_tag_remap.values())

        return related_box_ids, image_added_token_type_ids, text_token_type_ids


class VLEForVCRQA2RPipeline(Pipeline):

    def __init__(self, vle_processor, *args, **kwargs):        
        self.vle_processor = vle_processor
        self.vle_tokenizer = self.vle_processor.tokenizer
        self.GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Payton', 'Skyler', 'Frankie', 'Pat', 'Quinn']
        self.person_name_id = 0
        self.max_text_len = 80
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params, forward_params, postprocess_params = {}, {}, {}
        return preprocess_params, forward_params, postprocess_params

    def __call__(self, vcr_image_root: str, meta_inputs: dict, **kwargs):

        inputs = {"vcr_image_root": vcr_image_root, "meta_inputs": meta_inputs}
        results = super().__call__(inputs, **kwargs)
        return results

    def preprocess(self, inputs):
        model_inputs = self.vcr_qa2r_preprocess(inputs["vcr_image_root"], inputs["meta_inputs"])
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=1):
        logits = model_outputs["logits"]
        loss = model_outputs["loss"]
        preds = torch.argmax(logits, dim=-1, keepdim=True)

        return [{"score": score, "pred": pred} for score, pred in zip(logits, preds)]

    def vcr_qa2r_preprocess(self, vcr_image_root, data):
        image_fn = data["img_fn"]
        objects = data["objects"]
        metadata_fn = data["metadata_fn"]
        question = data["question"]
        answer_choices = data["answer_choices"]
        rationale_choices = data["rationale_choices"]
        answer_label = data["answer_label"]
        rationale_label = data["rationale_label"]

        question_text = question
        answer_text = answer_choices[answer_label]
        rationale_text = rationale_choices
        text_tokens, text_ids, obj_tags, text_raw = self.build_text(question_text, answer_text, rationale_text, objects, self.vle_tokenizer)
        encoding = [self.vle_tokenizer(
            ''.join(self.vle_tokenizer.convert_ids_to_tokens(text_ids_)),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        ) for text_ids_ in text_ids]

        obj_tags = [[-1] + tags + [-1] for tags in obj_tags]

        image = Image.open(os.path.join(vcr_image_root, image_fn))
        image_feature = self.vle_processor.image_processor(image)
        width = image.size[0]
        height = image.size[1]
        with open(os.path.join(vcr_image_root, metadata_fn), 'r') as f:
            vcr_image_metadata = json.load(f)
        boxes = vcr_image_metadata['boxes']
        patch_boxes = self.get_patch_box(boxes, width, height, self.model.config.vision_config.patch_size, self.model.config.vision_config.image_size)
        related_box_ids, image_added_token_type_ids, text_token_type_ids = list(zip(*[self.get_related_box_ids_and_token_type_ids(tags) for tags in obj_tags]))
        related_patch_boxes = [[patch_boxes[i] for i in related_box_ids[j]] for j in range(4)]

        processed_data = {
            "image": image_feature,
            "text": (text_raw, encoding),
            "obj_tags": obj_tags,
            "label": rationale_label,
            "patch_ids": [[patch_box[1] for patch_box in related_patch_boxes[i]] for i in range(4)],
            "extend_token_type_ids": [(image_added_token_type_ids[i], text_token_type_ids[i]) for i in range(4)],
        }

        model_inputs = {
            "input_ids": [[processed_data["text"][1][i]["input_ids"]] for i in range(4)],
            "attention_mask": [[processed_data["text"][1][i]["attention_mask"]] for i in range(4)],
            "token_type_ids": [[processed_data["text"][1][i]["token_type_ids"]] for i in range(4)],
            "pixel_values": torch.Tensor([processed_data["image"]["pixel_values"][0]]),
        }
        model_inputs = BatchEncoding(model_inputs, tensor_type='pt')
        model_inputs.update({
            "patch_ids": [[processed_data["patch_ids"][i]] for i in range(4)],
            "vcr_labels": [processed_data["label"]],
            "extend_token_type_ids": [list(zip(processed_data["extend_token_type_ids"][i])) for i in range(4)],
            "return_loss": True,
        })
        return model_inputs

    def retokenize_and_convert_to_ids_with_tag(self, raw_tokens, objects_replace_name, tokenizer, non_obj_tag=-1, add_space_b4_first_token=False):
        parsed_tokens = []
        tags = []
        align_ids = []
        raw = []
        align_id = 0
        for idx, mixed_token in enumerate(raw_tokens):
            if isinstance(mixed_token, list):
                tokens = [" " + objects_replace_name[o] for o in mixed_token]
                if idx == 0 and not add_space_b4_first_token:
                    tokens[0] = tokens[0].lstrip()
                retokenized_tokens = tokenizer.tokenize(tokens[0])
                raw.append(tokens[0])
                tags.extend([mixed_token[0] + non_obj_tag + 1 for _ in retokenized_tokens])
                align_ids.extend([align_id for _ in retokenized_tokens])
                align_id += 1
                for token, o in zip(tokens[1:], mixed_token[1:]):
                    retokenized_tokens.append(tokenizer.tokenize(' and')[0])
                    tags.append(non_obj_tag)
                    align_ids.append(align_id)
                    align_id += 1
                    re_tokens = tokenizer.tokenize(token)
                    retokenized_tokens.extend(re_tokens)
                    tags.extend([o + non_obj_tag + 1 for _ in re_tokens])
                    align_ids.extend([align_id for _ in re_tokens])
                    align_id += 1
                    raw.extend([' and', token])
                parsed_tokens.extend(retokenized_tokens)
            else:
                # fully align to original tokens
                if True in [unicodedata.category(str_) == 'Co' for str_ in mixed_token]:
                    continue
                if idx != 0 or add_space_b4_first_token:
                    mixed_token = " " + mixed_token
                raw.append(mixed_token)
                retokenized_tokens = tokenizer.tokenize(mixed_token)
                parsed_tokens.extend(retokenized_tokens)
                align_ids.extend([align_id for _ in retokenized_tokens])
                tags.extend([non_obj_tag for _ in retokenized_tokens])
                align_id += 1
        ids = tokenizer.convert_tokens_to_ids(parsed_tokens)
        ids_with_tag = list(zip(parsed_tokens, ids, tags, align_ids))
        
        return ids_with_tag, raw

    def build_text(self, question_text, answer_text, rationale_text, objects, tokenizer):
        objects_replace_name = []
        for o in objects:
            if o == 'person':
                objects_replace_name.append(self.GENDER_NEUTRAL_NAMES[self.person_name_id])
                self.person_name_id = (self.person_name_id + 1) % len(self.GENDER_NEUTRAL_NAMES)
            else:
                objects_replace_name.append(o)

        non_obj_tag = -1
        question_text, question_text_raw = self.retokenize_and_convert_to_ids_with_tag(question_text, objects_replace_name, tokenizer, non_obj_tag=non_obj_tag)
        answer_text, answer_text_raw = self.retokenize_and_convert_to_ids_with_tag(answer_text, objects_replace_name, tokenizer, non_obj_tag=non_obj_tag)
        rationale_text = [self.retokenize_and_convert_to_ids_with_tag(r_t, objects_replace_name, tokenizer, non_obj_tag=non_obj_tag) for r_t in rationale_text]
        rationale_text, rationale_text_raw = list(zip(*rationale_text))
        for r_t, r_t_raw in zip(rationale_text, rationale_text_raw):
            while len(question_text) + len(answer_text) + len(r_t) > self.max_text_len - 4:
                if len(r_t) > len(question_text) + len(answer_text):
                    r_t.pop()
                elif len(question_text) > 1:
                    question_text.pop()
                else:
                    answer_text.pop()
        
        text_tokens = [[q_t[0] for q_t in question_text] + [tokenizer.sep_token] + [a_t[0] for a_t in answer_text] + [tokenizer.sep_token] + [r_t_t[0] for r_t_t in r_t] for r_t in rationale_text]
        text_ids = [[q_t[1] for q_t in question_text] + [tokenizer.sep_token_id] + [a_t[1] for a_t in answer_text] + [tokenizer.sep_token_id] + [r_t_t[1] for r_t_t in r_t] for r_t in rationale_text]
        obj_tags = [[q_t[2] for q_t in question_text] + [-1] + [a_t[2] for a_t in answer_text] + [-1] + [r_t_t[2] for r_t_t in r_t] for r_t in rationale_text]
        text_raw = [question_text_raw + answer_text_raw + rationale_text_raw_ for rationale_text_raw_ in rationale_text_raw]
        
        return text_tokens, text_ids, obj_tags, text_raw

    def get_patch_box(self, boxes, width, height, patch_size, image_size):
        patch_count_w = image_size // patch_size
        patch_count_h = image_size // patch_size
        patch_width = width / patch_count_w
        patch_height = height / patch_count_h

        patch_boxes = []
        for box in boxes:
            box = box[:4]
            patch_x1 = int(box[0] // patch_width)
            patch_y1 = int(box[1] // patch_height)
            patch_x2 = int(box[2] // patch_width)
            patch_y2 = int(box[3] // patch_height)

            patch_x1 = patch_x1 if patch_x1 >= 0 else 0
            patch_y1 = patch_y1 if patch_y1 >= 0 else 0
            patch_x2 = patch_x2 + 1 if patch_x2 < patch_count_w else patch_count_w
            patch_y2 = patch_y2 + 1 if patch_y2 < patch_count_h else patch_count_h

            patch_box = [
                patch_x1 * patch_width,
                patch_y1 * patch_height,
                patch_x2 * patch_width,
                patch_y2 * patch_height
            ]

            patch_ids = [patch_count_w * y + x for y in range(patch_y1, patch_y2) for x in range(patch_x1, patch_x2)]
            patch_boxes.append([patch_box, patch_ids])

        return patch_boxes

    def get_related_box_ids_and_token_type_ids(self, obj_tags):
        no_obj_tag = -1
        obj_tags_set = set()
        for tag in obj_tags:
            if tag != no_obj_tag:
                obj_tags_set.add(tag)
        obj_tag_remap = {t: i + 2 for i, t in enumerate(obj_tags_set)}
        text_token_type_ids = [obj_tag_remap[tag] if tag != no_obj_tag else 0 for tag in obj_tags]
        related_box_ids = list(obj_tag_remap.keys())
        image_added_token_type_ids = list(obj_tag_remap.values())

        return related_box_ids, image_added_token_type_ids, text_token_type_ids