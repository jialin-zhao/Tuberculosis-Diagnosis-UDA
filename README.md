# Tuberculosis-Diagnosis-UDA

An unsupervised domain adaptation method for tuberculosis classification in CXR images.

## data

json
```bash
  - train
    - source
      - NM
      - TB
      - NTN
    - target
      - NM
      - TB
      - NTN
  - test
    - target
      - NM
      - TB
      - NTN

## usage

```bash
python main.py --config config/msdan.yaml --json_path data/mydata3.json 
