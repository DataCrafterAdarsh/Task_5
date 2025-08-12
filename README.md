# Task 5 — Decision Trees & Random Forests

**Objective:** Train and analyze Decision Tree & Random Forest models for binary classification using the Heart Disease dataset.  
**Dataset:** 1025 rows × 14 columns, no missing values, target = `target` (0/1).

---

## Steps Performed
1. **Data Check:** Verified shape, column types, and missing values (none).  
2. **Split Data:** Stratified train/test split (80/20).  
3. **Decision Tree:**  
   - Trained shallow tree (`max_depth=3`) for visualization.  
   - Evaluated overfitting by varying `max_depth` (1–20).  
   - Best accuracy at `max_depth=9` → Test Accuracy ≈ **0.9854**.  
4. **Random Forest:**  
   - 200 trees → Test Accuracy **1.0000**.  
5. **Feature Importances:**  
   - DT Top: `cp`, `ca`, `thal`, `age`.  
   - RF Top: `cp`, `thalach`, `ca`, `oldpeak`.  
6. **Cross-Validation (5-fold):**  
   - Decision Tree: Mean ≈ **0.9971**, Std ≈ 0.0059.  
   - Random Forest: Mean ≈ **0.9961**, Std ≈ 0.0078.

---

## Key Insights
- Both models perform extremely well — check for data leakage or label imbalance.  
- RF handles overfitting better and is slightly more stable.  
- No scaling needed; treat categorical-coded columns with care when interpreting splits.

---

## Files
- `README.md` (this file).
- `task5_notebook.ipynb` — Block-wise code execution.  
- Dataset CSV.  
- 'img(2)'
- 'img(3)'
- 'img(4)'
- 'img(5)'
- 'img1'

---

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib graphviz
python task5_script.py
