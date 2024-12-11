# Paper Analysis

This repository is designed to automatically analyze all files from a BibTeX file. It sends them to ChatGPT with a prompt to add additional information.

## Instructions for Use (`paperAnalysis.py`)

1. Set up `config.ini` with your API key.
2. Paste your Bibtex-File in the data folder.
3. Adjust the prompt. 
4. Run `paperAnalysis.py`.

```ini
# config.ini
[API]
key = your_api_key_here
```

```bash
python paperAnalysis.py
```

Make sure to replace `your_api_key_here` with your actual API key.

## Instructions for Use (`evaluationMethodAnalysis.py`)

1. Set up `config.ini` with your API key as descried above.
2. Place your Bibtex-File in the data folder (as `data.bib`).
3. Select the number of papers selected randomly from all papers in the Bibtex file by adjusting the `num_papers` variable in `evaluationMethodAnalysis.py`.
4. Run `evaluationMethodAnalysis.py`.