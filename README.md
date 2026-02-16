# Finance CSV Analyzer

A **no-database** trade CSV analyzer with a visual dashboard. Upload a CSV with trade data (Trade #, Type, Date and time, Price INR, Net P&L INR, Cumulative P&L INR). Data is **sorted by trade** (entry + exit = one trade), then ROI is shown **yearly** and **monthly** (with a **year dropdown** for monthly view). All in memory; nothing is stored.

## Features

- **Sort by trade** – Each trade has two rows (Entry and Exit). The table is sorted by Trade # then date so entry and exit are together and counted as one trade.
- **Sorted trades table** – First thing you see after upload: full CSV sorted by trade.
- **Yearly ROI** – ROI % per year from Cumulative P&L INR.
- **Monthly ROI** – ROI % per month with a **dropdown to select year** so you see only that year’s months (no mixing of all years).
- **Extra** – Total ROI, average yearly ROI, best/worst year, cumulative growth index, yearly volatility.
- **No database** – All processing is in-memory from the uploaded file.

## CSV format

Your CSV should have these columns (names can vary slightly):

| Column             | Description                                |
|--------------------|--------------------------------------------|
| **Trade #**        | Unique id for each trade (entry + exit = 1) |
| **Type**           | e.g. "Entry short", "Exit short"           |
| **Date and time**  | e.g. 21-01-2010 14:16 (day–month–year)    |
| **Signal**         | Optional                                  |
| **Price INR**      | Price in INR                              |
| **Net P&L INR**    | Net profit/loss for that leg               |
| **Cumulative P&L INR** | Running total P&L                     |

## Virtual environment and run

1. **Create a virtual environment**

   **Windows (PowerShell or CMD):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   (If `python` is not found, try `py -m venv venv` on Windows.)

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   python -m streamlit run app.py
   ```
   Or, if Streamlit is on your PATH: `streamlit run app.py`

4. Open the URL shown (e.g. http://localhost:8501), upload your CSV, and use:
   - **Sorted trades** table at the top
   - **Year range** slider in the sidebar for ROI
   - **Select year** dropdown for the monthly ROI view

## Project structure

- `app.py` – Streamlit dashboard (upload, sorted table, yearly/monthly ROI, year dropdown)
- `analyzer.py` – Load trades CSV, sort by trade, ROI from Cumulative P&L (no DB)
- `requirements.txt` – Python dependencies
- `sample_finance_data.csv` – Optional sample (simple date/value format; for full trade format use your own CSV)

No database or config files are required; everything runs from the uploaded CSV.
