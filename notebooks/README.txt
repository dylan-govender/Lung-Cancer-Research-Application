Google Drive Link to access all saved and downloaded models and evaluations:
https://drive.google.com/drive/folders/1BFMa-4MUvW7uteWN4fEOTuU5YBhGRVOO?usp=sharing

Deployed website for Lung Cancer Research. The software application that accompanies this research:
https://lung-cancer-research.streamlit.app/

PLEASE NOTE: The website will currently not predict an image. The Lung Cancer Image Analysis Section is down for now due to the large models that need deployment. 

The software application code can be found in my GitHub repository:
https://github.com/dylan-govender/Lung-Cancer-Research-Application

CODE for each Jupyter Notebook may not run due to different google drive accounts.
You can set up your google drive by uploading the datasets and then configuring the first part of the code where it says 1. Downloading Dependencies and Data

Change the Code Here by locating in Google Colab the datasets and copying the path:
PASTE the path in between !cp 'PASTE PATH HERE'

#@title **1.2 Downloading Data**

# for capturing output
%%capture

# Data
!cp '/content/drive/MyDrive/Honours Project/Datasets/Lung_Cancer_Histopathological_Dataset.zip' 'Lung_Cancer_Histopathological_Dataset.zip'
!unzip '/content/Lung_Cancer_Histopathological_Dataset.zip'
!rm '/content/Lung_Cancer_Histopathological_Dataset.zip'

If you retraining a ViT Model please use TPU or GPU.

DATASET LINKS
I have downloaded one dataset. The other is too large so please use the links.
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images
https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/data

NOTE: If some libraries may not be configured on your colab account. You may get an error. Please use the command !pip install library_name
