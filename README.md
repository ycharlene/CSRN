# Cross-modal Semantic Refinement Network (CSRN)
![image2](./images/fg2.jpg)

## Usage
We use single NVIDIA A800 80GB GPUs for training and evaluation.
```
torch 2.1.2+cu118
transformers 4.50.0
timm 1.0.15
accelerate 1.6.0
```
## Prepare Datasets

Download the [RSICD](https://github.com/201528014227051/RSICD_optimal), [RSITMD](https://github.com/xiaoyuan1996/AMFMN/tree/master/RSITMD) and [UCM-Captions](https://github.com/201528014227051/RSICD_optimal) datasets from the official websites or public repositories.

Organize them in `your dataset root dir` folder as follows:

```text
|-- your dataset root dir/
|   |
|   |-- <RSICD>/
|   |   |-- images/
|   |   |   |-- ...
|   |   |-- rsicd.json
|   |
|   |-- <RSITMD>/
|   |   |-- images/
|   |   |   |-- ...
|   |   |-- rsitmd.json
|   |
|   |-- <UCM>/
|   |   |-- images/
|   |   |   |-- ...
|   |   |-- ucm.json
```

## Training
```
python train.py
```
## Testing
```
python test.py
```

## Acknowledgements

Some components of this code implementation are adapted from [IRRA](https://github.com/anosorae/IRRA). We sincerely thank the authors for their excellent work and open-source contribution.
