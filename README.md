# Modality-Agnostic Adapter (MAA) For Fine-grained Scene Image Classification

This is the official repo of the paper **Fine-Grained Scene Image Classification with Modality-Agnostic Adapter** to appear in **ICME 2024**.

## Environment

```bash
conda env create -f environment.yml
conda activate maa
pip install -r requirements.txt

git clone https://github.com/matt-peters/allennlp.git
git clone https://github.com/allenai/kb.git

cd allennlp
git checkout 2d7ba1cb108428aaffe2dce875648253b44cb5ba
pip install -e .
cd ..

cd kb
pip install -r requirements.txt 
python -c "import nltk; nltk.download('wordnet')"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
pip install -e .
cd ..
```

## Dataset

### 1. Download datasets

Images of [ConText](https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html) and [Activity](https://github.com/MCLAB-OCR/KnowledgeMiningWithSceneText/blob/main/README_datasets.md) can be downloaded from the above links.

Annotation files and google ocr results can be downloaded [here](https://drive.google.com/file/d/1sWO9Ek0m-WfPmstY_ytWZQYsXNgDBnhb/view?usp=sharing).

### 2. Folder structure

```
datasets
├── activity
│   ├── images
│   ├── text.json
│   └── split_0.json
└── context
    ├── images
    ├── text.json
    └── split_0.json
```

## Usage

### 1. Train

```bash
python main.py -c CONFIG_PATH
```

For example:

```bash
python main.py -c configs/train_context.toml
python main.py -c configs/train_activity.toml
```

You can also pass parameters like this:

```bash
python main.py -c CONFIG_PATH --cfgs OUTPUT_DIR outputs NUM_EPOCHS 50 BATCH_SIZE_PER_GPU 8
```

The parameters after `--cfgs` are config items in `configs/*.toml`.

### 2. Test

```
python main.py -c TEST_CONFIG_PATH
```

## Acknowledgments

https://github.com/MCLAB-OCR/KnowledgeMiningWithSceneText

https://github.com/AndresPMD/Fine_Grained_Clf

https://github.com/AndresPMD/GCN_classification

https://github.com/allenai/kb

https://github.com/rwightman/pytorch-image-models
