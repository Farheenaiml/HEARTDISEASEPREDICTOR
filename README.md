# Heart Disease Predictor 🫀

This project is a **Heart Attack Prediction Web App** built using **Python and Streamlit**.  
It predicts whether a person is at risk of heart disease based on clinical input parameters.

You can use it in two ways:
1. **Use Pretrained Model (Recommended)** – directly run the app with the model already trained and saved as `heart_attack_model.pkl`.  
2. **Train Your Own Model** – use the provided dataset link and train the model in Google Colab, then save and use it locally.

---

## 📊 Dataset
The dataset used for training comes from:  
[Heart Disease Prediction Dataset](https://raw.githubusercontent.com/anshupandey/heart-disease-prediction/master/heart.csv)

---

## 📂 Project Structure
```
HEARTDISEASEPREDICTOR/
│── app.py                 # Streamlit web app
│── heart.py               # Model training / helper script (if needed)
│── run.py                 # Alternate script to run locally
│── heart_attack_model.pkl # Pretrained ML model
│── scaler.pkl             # Scaler for preprocessing
│── requirements.txt       # Python dependencies
│── .gitignore             # Ignored files (e.g., venv/)
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/HEARTDISEASEPREDICTOR.git
cd HEARTDISEASEPREDICTOR
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Run with Pretrained Model (Easy Way)
Simply start the Streamlit app:
```bash
streamlit run app.py
```
This will launch the website in your browser.  
The app automatically loads `heart_attack_model.pkl` and `scaler.pkl` to make predictions.

---

### Option 2: Train Your Own Model (From Dataset)
1. Open [Google Colab](https://colab.research.google.com/).  
2. Load the dataset from:  
   ```python
   url = "https://raw.githubusercontent.com/anshupandey/heart-disease-prediction/master/heart.csv"
   ```
3. Train the model using your notebook (`heart.py` script can guide you).  
4. Save the trained model as `heart_attack_model.pkl` and the scaler as `scaler.pkl`.  
5. Place them in the project folder and run the app again:
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Tech Stack
- **Python**
- **Scikit-learn**
- **Pandas & NumPy**
- **Streamlit** (for web interface)

---

## 👨‍💻 Author
Developed by Farheen Patel.  
Feel free to fork this repo, improve, and contribute!
