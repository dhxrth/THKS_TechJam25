# THKS_TechJam25

### Our project tackles the problem of assessing the quality and relevance of location-based reviews, where noisy or generic comments often overshadow useful feedback. We built an end-to-end pipeline that cleans raw reviews, fine-tunes a transformer model, and outputs reliable quality scores that can be used to rank reviews for businesses and consumers.

### Problem Statement Tackled

“How can we evaluate and enhance the quality and relevance of user-generated, location-based reviews so that businesses and consumers benefit from more trustworthy insights?”

### Our solution:

Cleans raw data with a custom preprocessing script (review_preprocess.py) that removes URLs, symbols, and stopwords but keeps negations to preserve sentiment.

Fine-tunes RoBERTa (FacebookAI/roberta-base) on labeled review data to distinguish high-quality reviews (specific, informative) from low-quality ones (generic, spammy).

Evaluates performance using accuracy, precision/recall, F1, and confusion matrices.

### Features & Functionality:

Preprocessing Engine – command-line tool for batch cleaning JSON/JSONL review datasets, outputting CSV/Parquet.

Custom Dataset Class – wraps reviews into a PyTorch Dataset with Hugging Face tokenization.

Transformer Classifier – RoBERTa model trained for multiclass classification of review quality.

Evaluation & Visualization – reports and plots generated using scikit-learn, matplotlib, seaborn.

### Libraries & Frameworks

Transformers (Hugging Face) – model, tokenizer, trainer.

PyTorch – dataset handling, GPU training.

scikit-learn – evaluation metrics.

pandas, numpy – data wrangling.

matplotlib, seaborn – visualization.

tqdm, chardet – progress bars and robust file handling.

### Assets & Datasets

Google Local Reviews dataset – for training and validation.

Manually labeled subset – for supervised fine-tuning.

Custom preprocessing outputs – cleaned text via review_preprocess.py

### How to run the code 
Preprocessing (review_preprocess.py)
Use this script to clean raw review datasets and output CSV files. 

For example:

python review_preprocess.py --in reviews.jsonl --out cleaned.csv --text-col text

Options include:

--in for the input file (.jsonl or .json)

--out for the output file (.csv or .parquet)

--text-col the column with raw text (default: text)

--out-col the column for cleaned text (default: text_clean)

--remove-numbers remove numeric tokens (optional)

--keep-negations / --no-keep-negations to control whether negations are preserved

Colab Notebook (THKS_TechJam.ipynb)
The notebook trains and evaluates the classification model.

Steps:

Open the notebook in Google Colab and enable GPU (Runtime → Change runtime type → GPU).

Run the first cell to install dependencies (transformers, torch, scikit-learn, matplotlib, seaborn, chardet).

Upload your preprocessed CSV from step 1.

Execute the notebook cells in order: data loading, train/test split, model setup (RoBERTa), fine-tuning, and evaluation.

Outputs include: model metrics (accuracy, F1, confusion matrix), visualizations, and a saved fine-tuned RoBERTa model.
