
# 🏠 Airbnb Recommendation System with LightFM

This repository contains an end-to-end hybrid recommendation system using the [LightFM](https://github.com/lyst/lightfm) framework, applied to Airbnb listing and user review data. The project implements a grid-searchable pipeline for model training, evaluation, and feature experimentation based on real-world Airbnb datasets.

---

## 📌 Project Overview

This recommender system aims to rank Airbnb listings for users based on historical booking behavior and listing metadata. We use a **hybrid recommendation approach** combining collaborative filtering with engineered content-based item features.

### Key Features:
- Time-aware data splitting per user
- Item-level feature engineering (room type, location, price, etc.)
- Configurable feature flags for ablation testing
- Early stopping with validation precision monitoring
- Evaluation using Precision@K, NDCG@K, MAP, AUC, and HitRate

---

## 🧱 Project Structure

| Step | Description |
|------|-------------|
| **1. Setup** | Mount Google Drive, install LightFM, import libraries |
| **2. Feature Engineering** | Generate interpretable item features from listing metadata |
| **3. Data Loading** | Clean review logs, apply interaction thresholds, align IDs |
| **4. Time-based Split** | Per-user chronological splitting into train/val/test |
| **5. Model Training** | LightFM training with early stopping |
| **6. Evaluation** | Top-K recommendation evaluation across metrics |
| **7. Grid Search** | Search across hyperparameters and feature sets |

---

## 📊 Best Performing Configuration

| Hyperparameter | Value |
|----------------|-------|
| Feature Flag   | `001010` (location + rating) |
| Latent Dimension | 32 |
| Learning Rate  | 0.03 |
| Loss Function  | WARP |
| Regularization | 1e-6 |

**Validation Precision@10**: 0.0314  
**Test AUC**: 0.9070

---

## 📈 Final Evaluation Results (Top-K @10)

| Metric         | Score   |
|----------------|---------|
| HR@10          | 0.0553  |
| NDCG@10        | 0.0194  |
| Precision@10   | 0.0077  |
| Recall@10      | 0.0273  |
| MAP            | 0.0116  |
| AUC            | 0.9070  |

---

## 🛠 Tech Stack

- Python, NumPy, Pandas
- LightFM, Scikit-learn, SciPy
- Google Colab
- Airbnb user and listing datasets

---

## 📁 Files

- `main.ipynb`: Full pipeline script with all steps
- **[Reviews.csv (Google Drive)](https://drive.google.com/file/d/1A-7ZIDiqLsvK57L79DhUkIdjRnjFACPs/view?usp=sharing)**: User-item interactions  
- **[final_listings.csv (Google Drive)](https://drive.google.com/file/d/1JW3cq3ApbxtksrnrNuCNU7hi6XxU-kVg/view?usp=sharing)**: Listing metadata with location, type, etc.


---

## 🙋 Contribution

This project was originally developed as part of the **BT4222 Mining Web Data for Business Insights** course at **National University of Singapore**.I contributed:

- Full pipeline implementation in LightFM
- Custom feature function generator and evaluation metrics
- Grid search logic with sparsity handling
- Interpretation and report writing

---

## 📬 Contact

For inquiries or collaboration, feel free to reach out:  
📧 emily21@korea.ac.kr 
