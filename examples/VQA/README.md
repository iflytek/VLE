# Fine-tuning on VQA

## Requirements

We use `Pytorch-Lightning` to fine-tuning the pre-trained VLEModel on VQA. To speedup training, we use `DeepSpeed`. The main packages are as follows:

```bash
pytorch_lightning==1.5.10
transformers==4.26.0
deepspeed==0.7.7
Pillow==8.1.0
tqdm==4.64.1
ipdb==0.13.4
numpy==1.21.6
einops==0.3.0
pyarrow==2.0.0
sacred==0.8.2
pandas==1.1.5
timm==0.4.12
ftfy
torchvision~=0.8.2
torch~=1.7.1
```

## Dataset Preparation for VQAv2

Download the VQAv2 dataset from [VQA official site](https://visualqa.org/download.html), including COCO [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip), [2015 test images](http://images.cocodataset.org/zips/test2015.zip), annotations ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)), and questions ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip)).

Please unzip and organize the dataset as follows:
    
    root
    ├── train2014            
    │   ├── COCO_train2014_000000000009.jpg                
    |   └── ...
    ├── val2014              
    |   ├── COCO_val2014_000000000042.jpg
    |   └── ...  
    ├── test2015              
    |   ├── COCO_test2015_000000000001.jpg
    |   └── ...         
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    ├── v2_OpenEnded_mscoco_val2014_questions.json
    ├── v2_OpenEnded_mscoco_test2015_questions.json
    ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
    ├── v2_mscoco_train2014_annotations.json
    └── v2_mscoco_val2014_annotations.json

We use `pyarrow` to serialize the datasets, the conversion script is `write_vqa.py`.

## Fine-tuning VLE on VQAv2

Hyperparameters for training are set in `vqa_train_config.json`.

Move the training related files to the same level of the directory as `models`, as follows:

    root
    ├── models
    │   └── VLE 
    |       └── ...
    ├── run_vqav2_ft.py
    ├── vqav2_datamodule.py
    └── vqav2_train_module.py

Specify the config file through `--train_config_file` and run the train script `run_vqav2_ft.py`. Here is an example:

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run_vqav2_ft.py --train_config_file=vqa_train_config.json
```

## Postprocess the checkpoint

After training, we convert the saved checkpoint, so that it can be loaded by `VLEModel`.

We first convert the deepspeed saved checkpoint to a pytorch checkpoint. The convert script is `zero_to_fp32.py`. If you didn't use `DeepSpeed` when training the model, this step could be skipped.

```bash
python zero_to_fp32.py <ckpt_dir> <output_file> <tag>
# for example:
python zero_to_fp32.py ./logs/VQAv2_seed0_from_vle-base-ft-vqa/version_0/checkpoints/epoch\=0-step\=0.ckpt step\=0.ckpt global_step0
```

Then, we convert the parameters' names to the same format as `VLEModel`. The convert script is `convert_checkpoint_after_ft.py`.
