Static Frontend for EV-Battery Health

1) Put files in a folder (frontend_static):
   - index.html
   - inference.html
   - styles.css
   - README.txt

2) Ensure your Flask backend is running (python app.py). Default expected Flask URL:
   http://127.0.0.1:8000

3) Open index.html in your browser (double-click). The dashboard will fetch:
   - Training plot: GET /api/results/plot/training
   - Parity plot:   GET /api/results/plot/parity_grid
   - Quick inference: POST /api/predict  (JSON body)

4) If Flask runs on another host/port, edit the API_BASE variable in the <script> sections of the HTML files.

Notes:
- The frontend locks battery_type to "Li-ion".
- This is a minimal static UI (no build step). If you want a more advanced React/Next UI later, I can generate that.
