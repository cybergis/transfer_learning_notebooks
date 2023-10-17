## Transfer Learning with Convolutional Neural Networks for Hydrological Streamline Detection

**Authors**: Nattapon Jaroenchai <sup>a, b</sup>, Shaowen Wang <sup>a, b, *</sup>, Lawrence V. Stanislawski <sup>c</sup>, Ethan Shavers <sup>c</sup>, E. Lynn Usery <sup>c</sup>, Shaohua Wang <sup>a, b</sup>, Sophie Wang, and Li Chen <sup>a, b</sup>  

*<sup>a</sup> Department of Geography and Geographic Information Science, University of Illinois at Urbana-Champaign, Urbana, IL, USA*  
*<sup>b</sup> CyberGIS Center for Advanced Digital and Spatial Studies, University of Illinois at Urbana-Champaign, Urbana, IL, USA*  
*<sup>c</sup> U.S. Geology Survey, Center of Excellence for Geospatial Information Science, Rolla, MO, USA*  
*<sup>d</sup> School of Geoscience and Info-Physics, Central South University, Changsha, Hunan, China*  

Last Updated Date: July 27, 2022

### Abstract 

Streamline network delineation is essential for various applications, such as agriculture sustainability, river dynamics, and watershed analysis. Machine learning methods have been applied for streamline delineation and have shown promising performance. However, performance drops substantially when a trained model is applied to different locations. This paper explores whether fine-tuning neural networks pre-trained on large-label datasets (e.g., ImageNet) can improve transferability from one geographic area to another. Specifically, we test transferability using small catchment streamlines from the Rowan County, NC, and Covington River, VA, areas in the eastern United States. First, we fine-tune eleven pre-trained U-Net models with various ResNet backbones in the Rowan County area and compare them with an attention U-net model trained from scratch on the same dataset. We find that the DenseNet169 model achieves an F1-score of 85%, about 4% higher than the attention U-net model. To compare the transferability of the models to a new geographic area, the three highest F1-score models from the Rowan County area are further fine-tuned with data in the Covington area. Similarly, we fine-tune the attention U-net model from the Rowan County area with the data in the Covington area. We find that fine-tuning the ResNet50 model achieves an F1-score of 65.58% in predicting the stream pixels in the Covington area, which is significantly higher than training the models from scratch in the Covington area or fine-tuning the attention U-net model from Rowan to Covington.

### Keywords: 

Transfer Learning, Convolutional neural network, Remote Sensing, Streamline detection

## How to use this repository

1. Open Google Colab (https://colab.research.google.com/)

2. Select GitHub as the notebook source and enter the URL below in the search. 

> https://github.com/cybergis/transfer_learning_notebooks

3. Download datasets from the URL below:

> https://drive.google.com/drive/folders/1VpHZcX4MRnt_3BUZmBnjldBb8DU18KV-

4. Put the downloaded folders in the samples folder. The folder structure should be 

|samples  
    --|rowancreek  
    --|covington   

5. Open the Main_Notebook.ipynb 


