# Phone Image Upload

A simple Windows desktop application that saves an image in a folder named
after a phone number and then shows the saved image.

## Setup

Open PowerShell in this folder and run:

```python -m venv .venvpowershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Saved images are placed inside:

```text
data/<phone-number>/<phone-number>_YYYY-MM-DD_HH-MM-SS_original.extension
```
