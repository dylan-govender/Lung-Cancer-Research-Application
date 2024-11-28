#  🫁 Lung Cancer Research Paper

***Abstract***-Lung cancer is a significant contributor to 
cancer-related mortality. With recent advancements in 
Computer Vision, Vision Transformers have gained traction 
and shown remarkable success in medical image analysis. This 
study explored the potential of Vision Transformer models (ViT, 
CVT, CCT ViT, Parallel ViT, Efficient ViT) compared to 
established state-of-the-art architectures (CNN) for lung 
cancer detection via medical imaging modalities, including CT 
and Histopathological scans. This work evaluated the impact of 
data availability and different training approaches on model 
performance. The training approaches included but were not 
limited to, Supervised Learning and Transfer Learning. 
Established evaluation metrics such as accuracy, recall, 
precision, F1-score, and area under the ROC curve (AUC
ROC) assessed model performance in terms of detection 
efficacy, data validity, and computational efficiency. ViT 
achieved an accuracy of 94% on a balanced dataset and an 
accuracy of 87% on an imbalanced dataset trained from the 
ground up. Cost-sensitive evaluation metrics, such as cost 
matrix and weighted loss, analysed model performance by 
considering the real-world implications of different types of 
errors, especially in cases where misdiagnosing a cancer case 
is far more critical.

#  🫁 Lung Cancer Research Application

A simple Streamlit app that is used for Lung Cancer Research. Uses Vision Transformer Models to predict on images.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lung-cancer-research.streamlit.app/)

---

## 🎥 Demo Video  

Check out the demo video showcasing the application's functionality:  

<p align="center">
  <a href="https://www.youtube.com/watch?v=nGwYNtuyl8E">
    <img src="https://img.youtube.com/vi/nGwYNtuyl8E/0.jpg" alt="Watch Demo Video" />
  </a>
</p>  

*(Click the image above or [here](https://www.youtube.com/watch?v=nGwYNtuyl8E) to watch the video.)*  

---

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
