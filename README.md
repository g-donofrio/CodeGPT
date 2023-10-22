# CodeGPT
ChatGPT enhanced with custom code.

## How to config
Create env file
```
cp .env.example .env
```
Set the variables in .env file

## How to run (Windows)
Create local environment:
```
python -m venv venv
```

Activate environment:
```
.\venv\scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the application
```
streamlit run ui.py
```