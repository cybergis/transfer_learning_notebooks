## Transfer Learning with Convolutional Neural Networks for Hydrological Streamline Detection

**Authors**: Nattapon Jaroenchai <sup>a, b</sup>, Shaowen Wang <sup>a, b, *</sup>, Lawrence V. Stanislawski <sup>c</sup>, Ethan Shavers <sup>c</sup>, E. Lynn Usery <sup>c</sup>, Shaohua Wang <sup>a, b</sup>, Sophie Wang, and Li Chen <sup>a, b</sup>  

*<sup>a</sup> Department of Geography and Geographic Information Science, University of Illinois at Urbana-Champaign, Urbana, IL, USA*  
*<sup>b</sup> CyberGIS Center for Advanced Digital and Spatial Studies, University of Illinois at Urbana-Champaign, Urbana, IL, USA*  
*<sup>c</sup> U.S. Geology Survey, Center of Excellence for Geospatial Information Science, Rolla, MO, USA*  
*<sup>d</sup> School of Geoscience and Info-Physics, Central South University, Changsha, Hunan, China*  

Last Updated Date: July 27, 2022

### Abstract 

Streamline network delineation plays a vital role in various scientific disciplines and business applications, such as agriculture sustainability, river dynamics, wetland inventory, watershed analysis, surface water management, and flood mapping. Traditionally, flow accumulation techniques have been used to extract streamlines, which delineate streamline primarily based on topological information. Recently, machine learning techniques such as the U-net model have been applied for streamlining delineation. Even though the model shows promising performance in geographic areas that it has been trained on, its performance drops significantly when applied to other areas. In this paper, we apply a transfer learning approach in which we use the pre-trained network architectures that have been trained on a large dataset, ImageNet. Then, we fine-tuned the neural networks using smaller datasets collected from Rowan Creek and Covington areas in the US. When we compared the models pre-trained on ImageNet with an attention U-net model which are fine-tuned on the Rowan Creek area, we found that the DenseNet169 model achieved an F1-score of 85% which is about 4% higher than the attention U-net model. Then, to compare the transferability of the models, the top three models in Rowan Creek area and the attention U-net model were fine-tuned further with the samples from the Covington area. We were able to achieve an F1-score of 71.87% in predicting the steam pixels in the Covington area which is significantly higher than training the model from scratch with the samples collected from the Covington area and slightly higher than the attention U-net model.

### Keywords: 

Transfer Learning, Convolutional neural network, Remote Sensing, Streamline detection

## How to use this repository

1. Open Google Colab (https://colab.research.google.com/)

2. Select GitHub as the source of the notebook and enter the URL below in the search. 

> https://github.com/cybergis/transfer_learning_notebooks

3. Download datasets from the URL below:

> https://drive.google.com/drive/folders/1VpHZcX4MRnt_3BUZmBnjldBb8DU18KV-

4. Put the downloaded folders in the samples folder. The folder structure should be 

|samples
    |--rowancreek
    |--covington 

5. Open the Main_Notebook.ipynb 


