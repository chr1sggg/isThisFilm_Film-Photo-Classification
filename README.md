# Film Photo Classifier
This model is built on the ResNet-18 CNN architecture and trained to distinguish film 
photographs from digital ones by analyzing differences in texture, color rendition, 
and overall visual characteristics. It designed to capture the subtle traits that make film 
photography unique, and achieves great classification performance.

# Demo

# Installation
1. Install libs

```
pip install huggingface_hub
pip install -r requirements.txt
```

2. Download Model

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="chr1sggg/Film-Photo-Classification", filename="model.pth")

```
3. Run

```
python main.py
```

