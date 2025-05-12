# Label Embedding for Self Attention

This repository is associated with the paper: [Multi-objective Representation for Numbers in Clinical Narratives: A CamemBERT-Bio-Based Alternative to Large-Scale LLMs](https://arxiv.org/abs/2405.18448).

## 📑 Table of Contents

- [⚙️ How to Set Up the Environment](#-how-to-set-up-the-environment)
- [🧑‍⚕️ Data Processing](#-data-processing)
- [🚀 Training CamemBert-Xval](#-training-camembert-xval)
- [📄 License](#-license)
- [📚 Citation](#-citation)

## ⚙️ How to Set Up the Environment

1. **Create a new Python virtual environment** named `xval_venv` using the `.yml` file by running the following command in your CLI:
    ```bash
    conda env create -f medical_xval/xval_venv.yml
    ```

2. **Activate the environment and create a Jupyter kernel** by running the following commands in the CLI:
    ```bash
    conda activate xval_venv
    pip install ipykernel
    python -m ipykernel install --user --name xval_venv --display-name "xval_venv"
    ```

3. **Replace the `transformers/camembert` folder** with our version present in this repo by running the following command in your CLI:
    ```bash
    scp -r medical_xval/camembert ~/.conda/envs/xval_venv/lib/python3.11/site-packages/transformers/models
    ```

4. **Replace the `transformers/__init__.py` file** with our version present in this repo:
    ```bash
    scp -r medical_xval/transformers__init__.py ~/.conda/envs/xval_venv/lib/python3.11/site-packages/transformers/__init__.py
    ```

## 🧑‍⚕️ Data Processing

Before running the developed models or executing the `train.ipynb` notebook, users must first preprocess the medical data.

1. Begin by creating a folder named `sadcsip`. Inside this folder, place the two private files:
   - `NLP_data/2020.06.03_CHUSJ_Data_PatientID.csv`
   - `NLP_data/Labelling Le - 0 to 100.csv`

   These files contain the medical notes along with their corresponding annotations.

2. Next, open and run all cells in the `data_preprocessing.ipynb` Jupyter notebook. While the notebook is lengthy, it is fully commented and should be easy to follow.

Once completed, the required datasets will be generated in the correct format within the `sadcsip` folder.

## 🚀 Training CamemBert-Xval

Now everything is ready to run the training process. Follow the instructions in the `Train.ipynb` notebook for detailed steps on training the LESABert model with the preprocessed data.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the original copyright and permission notice are included in all copies or substantial portions of the software.

## 📚 Citation

If you use this code or dataset in your research, please cite:

**Plain-text citation:**  
Lompo, A., Le, T.-D., Jouvet, P., & Noumeir, R. (2025). *medical\_token\_embedding: Preprocessing and Embedding of Medical Notes*. GitHub repository: https://github.com/sadc-lab/medical_token_embedding

**BibTeX:**
```bibtex
@misc{lompo2025medicaltokenembedding,
  author       = {Aser Lompo and Thanh-Dung Le and Philippe Jouvet and Rita Noumeir},
  title        = {medical_token_embedding: Preprocessing and Embedding of Medical Notes},
  year         = {2025},
  howpublished = {\url{https://github.com/sadc-lab/medical_token_embedding}},
}