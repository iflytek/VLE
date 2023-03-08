# Inference on Visual Commonsense Reasoning (VCR)

## Dataset Preparation for VCR

Download the VCR dataset from [VCR official site](https://visualcommonsense.com/download/), including [Annotations](https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1annots.zip) and [Images](https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip).
Unzip the downloaded file.

## Inference with VCR pipeline

Here are two examples of using fine-tuned VLEForVCR Q2A and QA2R models to infer on a VCR sample.
The sample's image is placed in `vcr_sample_images/` as follows, and the annotation `meta_data` is taken from VCR validation set.

    vcr_sample_images
    └──lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban
        ├──1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168_0.jpg
        └──1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168_0.json

### VCR Q2A
```python
from models.VLE import VLEForVCRQ2A, VLEProcessor, VLEForVCRQ2APipeline

model_name = 'vle-large-for-vcr-q2a'
model = VLEForVCRQ2A.from_pretrained(model_name)
vle_processor = VLEProcessor.from_pretrained(model_name)
vcr_q2a_pipeline = VLEForVCRQ2APipeline(model=model, device='cpu', vle_processor=vle_processor)

vcr_image_root = 'pics/vcr_sample_images'
meta_data = {"movie": "1054_Harry_Potter_and_the_prisoner_of_azkaban", "objects": ["person", "person", "person", "car", "cellphone", "clock"], "interesting_scores": [-1, 0], "answer_likelihood": "possible", "img_fn": "lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168@0.jpg", "metadata_fn": "lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168@0.json", "answer_orig": "No, 1 is a visitor.", "question_orig": "Does 1 live in this house?", "rationale_orig": "1 is wearing outerwear, holding an umbrella, and there is a car outside.", "question": ["Does", [0], "live", "in", "this", "house", "?"], "answer_match_iter": [2, 3, 0, 1], "answer_sources": [10104, 5332, 1, 16646], "answer_choices": [["No", ",", [0], "lives", "nowhere", "close", "."], ["Yes", ",", [0], "works", "there", "."], ["No", ",", [0], "is", "a", "visitor", "."], ["No", [1], "does", "not", "belong", "here", "."]], "answer_label": 2, "rationale_choices": [[[0], "is", "nicely", "dressed", "with", "a", "tie", ".", "people", "dress", "up", "when", "they", "visit", "someone", "else", "."], [[2], "sits", "comfortably", "in", "a", "chair", ",", "reading", "papers", ",", "while", "it", "seems", [0], "has", "just", "arrived", "and", "is", "settling", "in", "."], [[1], "is", "wearing", "a", "coat", "and", "muff", "and", "is", "sitting", "as", "if", "a", "visitor", "."], [[0], "is", "wearing", "outerwear", ",", "holding", "an", "umbrella", ",", "and", "there", "is", "a", "car", "outside", "."]], "rationale_sources": [26162, 12999, 6661, 1], "rationale_match_iter": [1, 3, 2, 0], "rationale_label": 3, "img_id": "val-0", "question_number": 1, "annot_id": "val-1", "match_fold": "val-0", "match_index": 1}

vcr_outputs = vcr_q2a_pipeline(vcr_image_root=vcr_image_root, meta_inputs=meta_data)
pred = vcr_outputs[0]["pred"]
print(f'Q: {meta_data["question"]}')
print(f'A1: {meta_data["answer_choices"][0]}')
print(f'A2: {meta_data["answer_choices"][1]}')
print(f'A3: {meta_data["answer_choices"][2]}')
print(f'A4: {meta_data["answer_choices"][3]}')
print(f'Label: {meta_data["answer_label"] + 1}')
print(f'predict: {pred[0] + 1}')
```

### VCR QA2R
```python
from models.VLE import VLEForVCRQA2R, VLEProcessor, VLEForVCRQA2RPipeline

model_name = 'vle-large-for-vcr-qa2r'
model = VLEForVCRQA2R.from_pretrained(model_name)
vle_processor = VLEProcessor.from_pretrained(model_name)
vcr_qa2r_pipeline = VLEForVCRQA2RPipeline(model=model, device='cpu', vle_processor=vle_processor)

vcr_image_root = 'pics/vcr_sample_images'
meta_data = {"movie": "1054_Harry_Potter_and_the_prisoner_of_azkaban", "objects": ["person", "person", "person", "car", "cellphone", "clock"], "interesting_scores": [-1, 0], "answer_likelihood": "possible", "img_fn": "lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168@0.jpg", "metadata_fn": "lsmdc_1054_Harry_Potter_and_the_prisoner_of_azkaban/1054_Harry_Potter_and_the_prisoner_of_azkaban_00.01.46.736-00.01.50.168@0.json", "answer_orig": "No, 1 is a visitor.", "question_orig": "Does 1 live in this house?", "rationale_orig": "1 is wearing outerwear, holding an umbrella, and there is a car outside.", "question": ["Does", [0], "live", "in", "this", "house", "?"], "answer_match_iter": [2, 3, 0, 1], "answer_sources": [10104, 5332, 1, 16646], "answer_choices": [["No", ",", [0], "lives", "nowhere", "close", "."], ["Yes", ",", [0], "works", "there", "."], ["No", ",", [0], "is", "a", "visitor", "."], ["No", [1], "does", "not", "belong", "here", "."]], "answer_label": 2, "rationale_choices": [[[0], "is", "nicely", "dressed", "with", "a", "tie", ".", "people", "dress", "up", "when", "they", "visit", "someone", "else", "."], [[2], "sits", "comfortably", "in", "a", "chair", ",", "reading", "papers", ",", "while", "it", "seems", [0], "has", "just", "arrived", "and", "is", "settling", "in", "."], [[1], "is", "wearing", "a", "coat", "and", "muff", "and", "is", "sitting", "as", "if", "a", "visitor", "."], [[0], "is", "wearing", "outerwear", ",", "holding", "an", "umbrella", ",", "and", "there", "is", "a", "car", "outside", "."]], "rationale_sources": [26162, 12999, 6661, 1], "rationale_match_iter": [1, 3, 2, 0], "rationale_label": 3, "img_id": "val-0", "question_number": 1, "annot_id": "val-1", "match_fold": "val-0", "match_index": 1}

vcr_outputs = vcr_qa2r_pipeline(vcr_image_root=vcr_image_root, meta_inputs=meta_data)
pred = vcr_outputs[0]["pred"]
print(f'Q: {meta_data["question"]}')
print(f'A: {meta_data["answer_choices"][meta_data["answer_label"]]}')
print(f'R1: {meta_data["rationale_choices"][0]}')
print(f'R2: {meta_data["rationale_choices"][1]}')
print(f'R3: {meta_data["rationale_choices"][2]}')
print(f'R4: {meta_data["rationale_choices"][3]}')
print(f'Label: {meta_data["rationale_label"] + 1}')
print(f'predict: {pred[0] + 1}')
```
