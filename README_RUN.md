Preprocessing and EDA for even-numbered files

Files operated on (even-numbered):
- `RawData/Dos-Drone/Dos2.csv`
- `RawData/Malfunction-Drone/Malfunction2.csv`
- `RawData/NormalFlight/Normal2.csv`

How to run (PowerShell):

```powershell
# create a virtual environment (optional)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\preprocess_even_files.py
```

Outputs will be in `output/` and processed scaled CSVs in `processed/`.
If any of the source CSVs are missing they will be skipped and noted in the summary.
