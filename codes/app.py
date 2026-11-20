import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/"result"
st.set_page_config(page_title="Fraud Defender",page_icon="🛡️",layout="wide")
st.markdown("""<style>.main{background:#0b1220}.stMetric{background:#111d31;padding:16px;border-radius:12px}.stMetric [data-testid="stMetricValue"],.stMetric [data-testid="stMetricValue"] *{color:#ffffff!important;opacity:1!important}.stSlider [data-testid="stThumbValue"],.stSlider [data-testid="stThumbValue"] *{color:#ffffff!important;opacity:1!important}</style>""",unsafe_allow_html=True)
st.title("🛡️ Fraud Defender")
st.caption("Rules → XGBoost → Human review → Feedback | Cost-aware decisions on anonymised card transactions")
model_path=OUT/"fraud_model.joblib"
if not model_path.exists():
 st.warning("Model not trained yet. In PowerShell run: Set-Location 'D:\\Fraud Dses\\code'; python -m pip install -r requirements.txt; python download_data.py; python train.py. Then refresh this page.")
 st.stop()
pack=joblib.load(model_path); model=pack["model"]; features=pack["features"]; threshold=pack["threshold"]
metrics=json.loads((OUT/"metrics.json").read_text())
with st.sidebar:
 st.header("Business costs")
 missed=st.slider("Missed fraud cost ($)",10,1000,150,10)
 blocked=st.slider("Genuine blocked cost ($)",1,100,10,1)
 otp=st.slider("OTP cost ($)",0,20,2,1)
 otp_cut=st.slider("OTP threshold",0.01,0.50,0.10,0.01)
 block_cut=st.slider("Block threshold",0.10,0.99,max(.65,float(threshold)),0.01)
 tab=st.radio("Demo",["Play the game","Cost lab","Rule discovery","Fraudster attack","Model report"])

def action(p,row=None):
 if row is not None and float(row["Amount"])>2000 and (float(row["Hour"])>=22 or float(row["Hour"])<5): return "BLOCK (RULE)"
 if p>=block_cut:return "BLOCK"
 if p>=otp_cut:return "ASK OTP"
 return "APPROVE"
def cost_actions(y, acts, amounts):
 total=0
 for yy,a,amt in zip(y,acts,amounts):
  total += (min(missed, float(amt)+20) if yy and a=="APPROVE" else (otp if yy and a.startswith("ASK OTP") else (0 if yy and a.startswith("BLOCK") else (blocked if not yy and a.startswith("BLOCK") else (otp if not yy and a.startswith("ASK OTP") else 0)))))
 return total
if tab=="Play the game":
 st.subheader("Judge vs AI: same 10 held-out transactions")
 X=pack["test_X"]; y=pack["test_y"]; probs=pack["test_p"]
 if "rounds" not in st.session_state: st.session_state.rounds=[]
 if st.button("Deal 10 transactions",type="primary"):
  rng=np.random.default_rng(); ids=rng.choice(len(X),10,replace=False)
  st.session_state.rounds=[(int(i),None) for i in ids]
 if st.session_state.rounds:
  rows=[]
  for j,(i,_) in enumerate(st.session_state.rounds):
   x=X.iloc[i]; p=float(probs[i]); ai=action(p,x)
   c1,c2,c3=st.columns([2,2,3]); c1.write(f"**#{j+1}** · ${x['Amount']:.2f}"); c2.write(f"AI risk {p:.1%} · **{ai}**")
   with c3: human=st.selectbox("Your decision",["APPROVE","ASK OTP","BLOCK"],key=f"human_{i}",label_visibility="collapsed")
   with st.expander(f"Why did the model score transaction #{j+1} this way?"):
    import xgboost as xgb
    contrib=model.get_booster().predict(xgb.DMatrix(x.to_frame().T[features]),pred_contribs=True)[0][:-1]
    top=np.argsort(np.abs(contrib))[-3:][::-1]
    st.write("Top score contributions (XGBoost TreeSHAP):", {features[k]:round(float(contrib[k]),4) for k in top})
   rows.append((i,human,ai,p))
  if st.button("Score this round"):
   ids=[r[0] for r in rows]; ys=y.iloc[ids].to_numpy(); amts=X.iloc[ids]["Amount"].to_numpy()
   hc=cost_actions(ys,[r[1] for r in rows],amts); ac=cost_actions(ys,[r[2] for r in rows],amts)
   st.session_state.rounds=[]
   a,b=st.columns(2); a.metric("Your estimated loss",f"${hc:,.0f}"); b.metric("AI estimated loss",f"${ac:,.0f}")
   st.caption("Teaching simulation uses test labels to score actions after your decisions. Cost values are assumptions.")
elif tab=="Cost lab":
 st.subheader("Business costs change the best action")
 X=pack["test_X"]; y=pack["test_y"].to_numpy(); p=np.asarray(pack["test_p"]); amounts=X["Amount"].to_numpy()
 ts=np.linspace(.01,.99,99); losses=[]
 for t in ts:
  acts=np.where(p>=t,"BLOCK","APPROVE"); losses.append(cost_actions(y,acts,amounts))
 opt=float(ts[int(np.argmin(losses))]); st.metric("Lowest-loss single threshold on test set",f"{opt:.2f}",f"${min(losses):,.0f} estimated loss")
 st.line_chart(pd.DataFrame({"Threshold":ts,"Estimated loss ($)":losses}).set_index("Threshold"))
 st.caption("Threshold curve is illustrative and uses held-out labels for analysis; do not use it as production model selection.")
elif tab=="Rule discovery":
 st.subheader("Readable rule sketch from model patterns")
 imp=pd.Series(model.feature_importances_,index=features).sort_values(ascending=False).head(8)
 st.bar_chart(imp)
 st.markdown(f"1. **IF** {imp.index[0]} is unusually high **THEN** review more closely.\n2. **IF** {imp.index[1]} is unusual **AND** amount is high **THEN** request OTP.\n3. **ELSE** use the model risk score and cost thresholds.")
 st.caption("These are teaching heuristics from feature importance, not verified decision-tree rules. Features V1–V28 are anonymised PCA components.")
elif tab=="Fraudster attack":
 st.subheader("Simplified adversarial evasion demonstration")
 p=np.asarray(pack["test_p"]); y=pack["test_y"].to_numpy(); mask=y==1
 evasion=st.slider("Fraud risk reduction (simulated disguise)",0.0,.95,.50,.05)
 baseline=float(np.mean(p[mask]>=block_cut)) if mask.any() else 0
 attacked=float(np.mean(p[mask]*(1-evasion)>=block_cut)) if mask.any() else 0
 a,b=st.columns(2); a.metric("Fraud caught before",f"{baseline:.1%}"); b.metric("After disguise",f"{attacked:.1%}",f"{attacked-baseline:+.1%}")
 st.info("Retraining is available from the training script after adding newly labelled examples. This slider changes scores only to illustrate evasion; it does not create real new fraud data.")
elif tab=="Model report":
 st.subheader("Held-out model evaluation")
 cols=st.columns(4)
 for c,k,label,fmt in zip(cols,["pr_auc","roc_auc","precision","recall"],["PR-AUC","ROC-AUC","Precision","Recall"],[".3f"]*4):c.metric(label,f"{metrics[k]:.3f}")
 st.write(f"Rows: {metrics['rows']:,} · Fraud cases: {metrics['frauds']:,} ({metrics['fraud_rate']:.3%}) · Test rows: {metrics['test_rows']:,} · threshold: {metrics['threshold']:.3f}")
 st.write("Confusion matrix [[TN, FP], [FN, TP]]:",metrics["confusion_matrix"])
 st.warning("Prototype, not for real card decisions. Random split may not reflect future drift; cost assumptions are illustrative.")





