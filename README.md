# Employee-Salary-Prediction-System
This repository contains an FCIS AI team project for employee salary classification.
The project was originally developed in a notebook, then adapted into a runnable Flask web application with:

project documentation
single-record model testing
batch CSV prediction
comparison between all trained models
The application predicts whether a record belongs to:

<=50K
>50K
Project Structure
app.py: Flask web application
salary_pipeline.py: preprocessing, training, prediction, and documentation helpers
train_model.py: script for training and saving model artifacts
Train.csv: training dataset
Test.csv: testing dataset
artifacts/: saved trained models and metrics
templates/: HTML pages
static/: extracted report figures used in the documentation page
AI_salary_prediction (1).ipynb: original notebook
Requirements
Recommended environment:

Python 3.9
pip
internet access during the first installation
Python packages:

flask
joblib
lightgbm
numpy
pandas
scikit-learn
xgboost
1. Download the Project
Clone the repository:

git clone <your-repository-url>
cd <your-repository-folder>
Or download it as a ZIP file from GitHub and extract it.

2. Create a Virtual Environment
It is better to run the project inside a virtual environment.

On macOS or Linux:

python3 -m venv .venv
source .venv/bin/activate
On Windows Command Prompt:

python -m venv .venv
.venv\Scripts\activate
On Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1
3. Install Dependencies
Install the required Python packages:

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
On Windows, replace python3 with python if needed.

4. macOS Note for LightGBM
If you are using macOS, lightgbm may require OpenMP.

Install it with Homebrew:

brew install libomp
If lightgbm raises a libomp.dylib error, this step is required.

5. Make Sure the Data Files Exist
The following files must be present in the project root:

Train.csv
Test.csv
Do not move them into another folder unless you also update the paths in salary_pipeline.py.

6. Train the Models
Run:

python3 train_model.py
This step will:

load Train.csv and Test.csv
preprocess the data
train all configured models
compare their performance
save the artifacts in artifacts/
Expected generated files:

artifacts/salary_model.joblib
artifacts/salary_models.joblib
artifacts/metrics.json
If these files already exist, you can still retrain to refresh them.

7. Start the Web Application
Run:

python3 app.py
Then open your browser and go to:

http://127.0.0.1:5000
8. Available Pages
The application contains the following pages:

Home: project introduction
Documentation: report-style academic explanation with figures from the notebook
Model Testing: single-record prediction and batch CSV testing
Technical Summary: model metrics and team information
9. How to Use the Model Testing Page
Single-record prediction
Open the Model Testing page.
Select values from the dropdown fields.
Click Run Prediction.
Review:
the best-model prediction
confidence
per-model comparison table
Batch CSV prediction
Prepare a CSV file.
Upload it from the Batch CSV Evaluation section.
Click:
Preview Results to preview predictions in the browser
Download Output CSV to download the predicted file
10. Accepted CSV Schemas
The app accepts either the train-style column names:

work-class
work-fnl
position
or the test-style names from the notebook:

workclass
fnlwgt
occupation
The rest of the expected columns should still match the original dataset structure.

11. If the App Does Not Show New Changes
If you edit the code and the browser still shows old behavior:

stop the running Flask process with Ctrl+C
start it again:
python3 app.py
Then refresh the browser.

12. Common Problems
Problem: ModuleNotFoundError
Install dependencies again:

python3 -m pip install -r requirements.txt
Problem: libomp.dylib error on macOS
Run:

brew install libomp
Problem: page opens but predictions fail
Check that:

Train.csv exists
Test.csv exists
the artifacts were created by running:
python3 train_model.py
Problem: model results look outdated
Retrain the project:

python3 train_model.py
Then restart the Flask app:

python3 app.py
13. Recommended Run Order on a New Laptop
Use this order on another machine:

git clone <your-repository-url>
cd <your-repository-folder>
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 train_model.py
python3 app.py
For macOS, add this before training if needed:

brew install libomp
14. Notes
The notebook remains in the repository for reference.
The web app is the recommended way to demonstrate the final project.
The saved model artifacts may change after retraining because the project compares multiple models and saves the best-performing one.
