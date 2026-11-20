# Fraud Defender — quick start

## Project contents
- `code/`: download, training and Streamlit demo
- `result/`: trained model, evaluation metrics and held-out predictions
- `paper/`: project report

## Run on Windows (PowerShell)
```powershell
cd 'D:\Fraud Dses\code'
python -m pip install -r requirements.txt
python download_data.py
python train.py
python -m streamlit run app.py
```
The Kaggle download may ask for Kaggle credentials. If that happens, accept the Kaggle terms and configure the Kaggle API token (`kaggle.json`) as described by Kaggle, then rerun `python download_data.py`. Dataset page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Demo sequence (3 minutes)
1. Model report: explain why PR-AUC and recall matter when fraud is ~0.17%.
2. Play the game: deal 10 unseen held-out transactions, choose actions, score estimated costs.
3. Cost lab: show the threshold/loss curve and explain costs are assumptions.
4. Rule discovery: discuss readable teaching heuristics and anonymised PCA features.
5. Fraudster attack: move the disguise slider to show the arms-race idea.

Model: XGBoost with `scale_pos_weight`, stratified 75/25 split, random seed 42. Action/cost logic is educational. Never use this prototype to approve or decline live transactions.

