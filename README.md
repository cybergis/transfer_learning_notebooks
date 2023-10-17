# Transfer Learning with Convolutional Neural Networks for Hydrological Streamline Detection

## Authors:

- **Nattapon Jaroenchai** (a, b)
- **Shaowen Wang** (a, b, *)
- **Lawrence V. Stanislawski** (c)
- **Ethan Shavers** (c)
- **E. Lynn Usery** (c)
- **Shaohua Wang** (a, b)
- **Sophie Wang**
- **Li Chen** (a, b)

### Affiliations:

- (a) Department of Geography and Geographic Information Science, University of Illinois at Urbana-Champaign, Urbana, IL, USA
- (b) CyberGIS Center for Advanced Digital and Spatial Studies, University of Illinois at Urbana-Champaign, Urbana, IL, USA
- (c) U.S. Geology Survey, Center of Excellence for Geospatial Information Science, Rolla, MO, USA
- (d) School of Geoscience and Info-Physics, Central South University, Changsha, Hunan, China

**Last Updated**: July, 2023

## Abstract:

Streamline network delineation is essential for various applications, such as agriculture sustainability, river dynamics, and watershed analysis. Machine learning methods have been applied for streamline delineation and have shown promising performance. However, performance drops substantially when a trained model is applied to different locations. In this paper, we explore whether fine-tuning neural networks that have been pre-trained on large label datasets (e.g., ImageNet) can improve transferability from one geographic area to another. Specifically, we test transferability using small catchment stream lines from the Rowan County, NC and Covington River, VA areas in the eastern United States. First, we fine-tune eleven pre-trained U-Net models with various ResNet backbones on the Rowan County area and compare them with an attention U-net model that is trained from scratch on the same dataset. We find that the DenseNet169 model achieves an F1-score of 85% which is about 4% higher than the attention U-net model. To compare the transferability of the models to a new geographic area, the three highest F1-score models from the Rowan County area are further fine-tuned with data in the Covington area. Similarly, we fine-tune the attention U-net model from the Rowan County area with the data in the Covington area. We find that fine-tuning ResNet50 model achieves an F1-score of 65.58% in predicting the stream pixels in the Covington area, which is significantly higher than training the models from scratch in the Covington area or fine-tuning attention U-net model from Rowan to Covington.

## Keywords:

- Transfer Learning
- Convolutional Neural Network
- Remote Sensing
- Streamline Detection

## Repository Usage:

1. **Google Colab Setup**:
    - Visit [Google Colab](https://colab.research.google.com/)
    - Choose "GitHub" as the notebook source.
    - In the search bar, enter the following repository URL: 
      ```
      https://github.com/cybergis/transfer_learning_notebooks
      ```

2. **Dataset Download and Configuration**:
    - Access and download the datasets from the following link:
      [Dataset Download](https://drive.google.com/drive/folders/1VpHZcX4MRnt_3BUZmBnjldBb8DU18KV-)
    - Ensure the datasets are placed in the 'samples' directory, maintaining the following structure:
      ```
      samples/
      ├── rowancreek/
      └── covington/
      ```

3. **Launching the Main Notebook**:
    - Navigate to and open `Main_Notebook.ipynb`.

4. **Execution**:
    - Follow the instructions and code snippets in the notebook for analysis and results.

We encourage researchers and enthusiasts to explore, adapt, and build upon this work. For any queries, feel free to raise an issue on the repository or contact the authors directly.
