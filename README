# Cross-Lingual Transferability of Training Data Extraction Attacks to Recover Memorized PII

This repository contains code and analysis notebooks for reproducing results from the paper **“Cross-Lingual Transferability of Training Data Extraction Attacks to Recover Memorized PII.”**

## Repository contents

- `01-Attacks_withActivations.ipynb`: training-data extraction experiments with activation analysis.
- `01-AttacksTransfer_withActivations.ipynb`: cross-lingual transfer experiments with activation analysis.
- `02-AttacksParaph_withActivations.ipynb`: paraphrase-based extraction experiments with activation analysis.
- `03-Attacks_Results.ipynb`: results analysis and visualization.
- `04-BridgeAnalysis-Paraphrases.ipynb`: bridge analysis comparing strict translations and paraphrases.
- `translate_dataset_local_fixed_priority_latest.py`: local model-based dataset translation.
- `web-novelty.py` and `web-novelty-para.py`: web search and provenance/novelty pipelines for the standard and paraphrase data configurations.
- `data/` and `data_p/`: local dataset roots used by the scripts and notebooks.

## Data access

The dataset will be shared for research purposes only. To request access, please email the corresponding author. The data must not be redistributed or used outside the approved research purpose.

## Setup

Use a Python environment with Jupyter installed to run the notebooks. The experiments use PyTorch, Transformers, Hugging Face Datasets, pandas, and other libraries imported in the notebooks. Install a PyTorch build compatible with your hardware and CUDA setup where applicable.

The web-novelty scripts have a dedicated dependency list:

```powershell
python -m pip install -r web_novelty_requirements.txt
```

Create an `env.json` file in the repository root for the Hugging Face and Tavily credentials required by the workflows. `env.json` is ignored by Git; keep credentials private and do not commit them.

```json
{
  "HF_TOKEN_TRANSLATION": "your-hugging-face-token",
  "HF_TOKEN_ATTACKS_TRANSFER": "your-hugging-face-token",
  "HF_TOKEN_ATTACKS": "your-hugging-face-token",
  "HF_TOKEN_ATTACKS_PARAPH": "your-hugging-face-token",
  "TAVILY_API_KEY": "your-tavily-api-key",
  "TAVILY_API_KEY_PARA": "your-tavily-api-key"
}
```

The Hugging Face token must have access to the model used by the selected workflow. Tavily credentials are needed only for the web-novelty scripts.

## Reproducing the analyses

1. Request and place the approved data in the expected local data directories.
2. Configure the model, language, PII type, and input/output paths in the relevant notebook or at the top of the translation script.
3. Run the corresponding attack notebook from top to bottom. Run `03-Attacks_Results.ipynb` and `04-BridgeAnalysis-Paraphrases.ipynb` after generating the experiment outputs they analyze.
4. To run the web-novelty workflows from the repository root:

	```powershell
	python web-novelty.py
	python web-novelty-para.py
	```

	Adjust each script's configuration block as needed before running. These workflows make external web/API requests.

5. To run local translation, review the configuration near the top of `translate_dataset_local_fixed_priority_latest.py`, then run:

	```powershell
	python translate_dataset_local_fixed_priority_latest.py
	```

The translation and extraction workflows can require substantial GPU memory and runtime. Exact results may depend on the model revision, hardware, library versions, and external search results.

