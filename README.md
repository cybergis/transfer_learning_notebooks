# Transfer Learning with Convolutional Neural Networks for Hydrological Streamline Detection

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
      [Dataset Download](10.6084/m9.figshare.24512698)
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


## Repository File Structure

## Repository File Structure

| Type      | Name                         | Description                                                                                          |
|-----------|------------------------------|------------------------------------------------------------------------------------------------------|
| Directory | `libs/`                      |                                                                                                      |
| File      | `libs/keras.py`              | Modified `keras.py` for fixing a bug in segmentation model library.                                  |
| Directory | `notebook_data/`             | Directory for data used within notebooks.                                                            |
| Directory | `samples/`                   | Sample data directory.                                                                               |
| File      | `samples/README.md`          | Instructions for data downloading and their structure.                                               |
| File      | `Main_Notebook.ipynb`        | The primary Jupyter notebook for this project.                                                       |
| File      | `README.md`                  | The main README for this repository.                                                                 |
| File      | `Traditional_Methods.ipynb`  | Notebook focusing on traditional methods of streamline detection.                                    |
| File      | `Train_Models.ipynb`         | Notebook to train the models in all scenarios.                                                       |
| File      | `unet_util.py`               | Utility functions specific to the U-Net model.                                                       |
