
Project: ML models separated and deployable via Streamlit

Structure:
- classification/: scripts to train two classification models and save joblib files
- regression/: scripts to train two regression models and save joblib files
- streamlit_app/app.py: app to load chosen saved models and run predictions
- data/: place your pima-indians-diabetes.csv and PRSA_data_2010.1.1-2014.12.31.csv here

Usage:
1. datasets are use by URL
2. Run trainers:
   python classification/model_A_kfold.py
   python classification/model_B_loocv.py
   python regression/model_A_split.py
   python regression/model_B_repeated.py
3. Choose the best model files (the scripts save them; compare printed metrics).
4. Start the Streamlit app:
   pip install -r requirements.txt
   streamlit run streamlit_app/app.py
