# Run the AI Study Tip Generator

## Get the project

Clone the GitHub repository or download and extract its ZIP file. Open a terminal in the project root (the folder containing `app.py` and `requirements.txt`). The commands below use relative paths, so they do not depend on where the project was downloaded.

Python 3.11 is recommended. The first run needs internet access to download GPT-2's base weights (about 550 MB) into `.cache/huggingface` inside this project folder. Model downloads and runtime caches are kept under the project-local `.cache` directory. The repository includes only the approximately 14.2 MB fine-tuned last-block checkpoint.

## Windows Command Prompt

Run these commands from the project root:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
python train.py --evaluate
python -m streamlit run app.py
```

## Windows PowerShell

Run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
python train.py --evaluate
python -m streamlit run app.py
```

If PowerShell blocks activation, use the virtual-environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe train.py --evaluate
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## macOS or Linux

Run these commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
python train.py --evaluate
python -m streamlit run app.py
```

## Open the app

When Streamlit starts, open the local URL printed in the terminal. The default is:

```text
http://localhost:8501
```

Enter a topic such as **Python Loops**, **Java OOP**, or **Machine Learning**, then click **Generate study tip**. Known topics use a curated example; if a GPT-2 draft is vague or unrelated, the app provides a practical study prompt instead.

## Running again

After setup, open a terminal in the project root, activate `.venv` using the command for your operating system above, and run:

```text
python -m streamlit run app.py
```

Training is only needed for a fresh setup or when changing the training data. Run `python train.py --evaluate` whenever you want to regenerate held-out outputs in `lab_outputs/evaluation_results.csv`. Press **Ctrl+C** in the terminal to stop Streamlit.
