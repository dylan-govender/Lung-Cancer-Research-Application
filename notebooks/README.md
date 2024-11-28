# Lung Cancer Research Application  

This repository accompanies the **Lung Cancer Research** project and includes:  
- A software application for lung cancer research and analysis.  
- Access to models, evaluations, and datasets for reproduction and further exploration.  

---

## 📂 **Google Drive Link**  
Access all saved and downloaded models, evaluations, and supplementary resources via the Google Drive link below:  
🔗 [Google Drive: Lung Cancer Models & Evaluations](https://drive.google.com/drive/folders/1BFMa-4MUvW7uteWN4fEOTuU5YBhGRVOO?usp=sharing)  

---

## 🌐 **Deployed Application**  
Explore the deployed website for Lung Cancer Research using the link below:  
🔗 [Lung Cancer Research Application](https://lung-cancer-research.streamlit.app/)  

**Please Note:**  
- The "Lung Cancer Image Analysis" section is currently **down** due to the size of the models that require deployment.  
- The website does not predict images at the moment.  

---

## 🛠️ **Code Repository**  
The code for this software application can be found in my GitHub repository:  
🔗 [GitHub: Lung Cancer Research Application](https://github.com/dylan-govender/Lung-Cancer-Research-Application)  

---

## ⚠️ **Important Notes**  

### **Running the Code in Jupyter Notebooks**
1. The Jupyter Notebook code may not work as-is because it uses **Google Drive paths** specific to the original development environment.  
2. To make the code functional:  
   - Upload the datasets to your **Google Drive**.  
   - Edit the code section labeled **"1. Downloading Dependencies and Data"** to match your dataset path.  

#### Example:
Replace `PASTE PATH HERE` in the code snippet with your dataset's path in Google Drive:  
```python
#@title **1.2 Downloading Data**

# for capturing output
%%capture

# Data
!cp 'PASTE PATH HERE' 'Lung_Cancer_Histopathological_Dataset.zip'
!unzip 'Lung_Cancer_Histopathological_Dataset.zip'
!rm 'Lung_Cancer_Histopathological_Dataset.zip'
