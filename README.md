# Tuberculosis-Diagnosis-UDA

An unsupervised domain adaptation method for tuberculosis classification in CXR images.

![framework](/images/resunet-tbroute.png)

## data

- DRRs generation: We employ a parallel projection model [1] to generate DRR images from CT images.

- The model reads the dataset in the format of a JSON file
  ```bash
  - train
    - source
      - NM (normal)
      - TB (tuberculosis)
      - NTN (other lung diseases)
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

- Collect the data you need and store it in the `/data`, and then create a JSON file

- Train a model:
  ```bash
  python main.py --config config/msdan.yaml --json_path data/yourDataJsonFile.json
  
## Citation

If you use this code for your research, please cite our papers.
