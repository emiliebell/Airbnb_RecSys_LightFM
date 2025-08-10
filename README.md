
# 🏠 Airbnb Recommendation System with LightFM

End-to-end **hybrid recommendation system** using [LightFM](https://github.com/lyst/lightfm), applied to Airbnb-style user booking data.  
Implements **time-aware splits**, **configurable content features**, **early stopping**, and **grid search** — producing a fully reproducible, research-grade pipeline.

---

## 📌 Why This Project Stands Out
- **Realistic Evaluation**: Per-user chronological split to prevent leakage
- **Hybrid Modeling**: Collaborative filtering + engineered item features
- **Reproducible Experiments**: Clear data schema, hyperparameter search, early stopping
- **Multi-metric Assessment**: Precision@K, NDCG, MAP, HitRate, AUC

---

## 🔧 Environment
- Python `3.11`, `lightfm==1.17`
- Google Colab (Drive mounted)
```python
from google.colab import drive; drive.mount('/content/drive')
!pip install lightfm==1.17
````

---

## 📁 Data

* **[Reviews.csv (Google Drive)](https://drive.google.com/file/d/1A-7ZIDiqLsvK57L79DhUkIdjRnjFACPs/view?usp=sharing)** – user-item interactions (`item_id`, `user_id`, `date`, `review_id` → `booked`=1 if review exists)
* **[final\_listings.csv (Google Drive)](https://drive.google.com/file/d/1JW3cq3ApbxtksrnrNuCNU7hi6XxU-kVg/view?usp=sharing)** – listing metadata (`room_type`, `property_type`, `loc_cluster`, `price`, `review_scores_rating`, `accommodates`)
  Filters: `min_user=10`, `min_item=20`

---

## 📊 Best Configuration

| Feature Flag            | Latent Dim | LR   | Loss | Reg  | Val P\@10 |
| ----------------------- | ---------- | ---- | ---- | ---- | --------- |
| `001010` (loc + rating) | 32         | 0.03 | WARP | 1e-6 | 0.0314    |

---

## 📈 Final Test Results (@10)

| HR     | NDCG   | Precision | Recall | MAP    | AUC    |
| ------ | ------ | --------- | ------ | ------ | ------ |
| 0.0553 | 0.0194 | 0.0077    | 0.0273 | 0.0116 | 0.9070 |

**Interpretation**

* **AUC 0.9070** → strong global ranking ability
* **Low P\@K** → sparsity (\~0.001) + minimal features limit top-rank precision

---

## 🙋 My Contributions

* Designed & implemented **full LightFM pipeline** (data prep → training → evaluation)
* Built **feature flag system** for interpretable item features
* Developed **grid search** with sparsity checks and early stopping
* Wrote **analysis & interpretation** connecting results to data constraints

---

## 🚀 Future Directions

* Add richer features (price bins, property type, text/image embeddings)
* Test alternative losses (BPR, logistic) + re-ranking (LambdaMART)
* Model temporal popularity shifts

---

📧 **[emily21@korea.ac.kr](mailto:emily21@korea.ac.kr)**
