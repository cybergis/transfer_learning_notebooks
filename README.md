<h1 style="text-align:center;line-height:1.5em;font-size:30px;">Transfer Learning with a Convolutional Neural Network for Hydrological Streamline Detection</h1>

Nattapon Jaroenchai(a, b), Shaowen Wang (a, b, *), Lawrence V. Stanislawski (c), Ethan Shavers (c), E. Lynn Usery (c), Shaohua Wang (a, b), Sophie Wang, and Li Chen (a, b)

a Department of Geography and Geographic Information Science, University of Illinois at Urbana-Champaign, Urbana, IL, USA  
b CyberGIS Center for Advanced Digital and Spatial Studies, University of Illinois at Urbana-Champaign, Urbana, IL, USA  
c U.S. Geology Survey, Center of Excellence for Geospatial Information Science, Rolla, MO, USA  
d School of Geoscience and Info-Physics, Central South University, Changsha, Hunan, China  


## Abstract

Streamline network delineation plays a vital role in various scientific disciplines and business applications, such as agriculture suitability, river dynamics, wetland inventory, watershed analysis, surface water survey and management, flood mapping. Traditionally, flow accumulation techniques have been used to extract streamline, which delineates the streamline solely based on topological information. Recently, machine learning techniques have created a new method for streamline detection using the U-net model. However, the model performance significantly drops when used to delineate streamline in a different area than the area it was trained on. In this paper, we examine the usage of transfer learning techniques, transfer the knowledge from the prior area and use the knowledge of the prior area as the starting point of the model training for the target area. We also tested transfer learning methods with different scenarios, change the input data by adding the NAIP dataset, retrain the lower and the higher part of the network, and varying sample sizes. We use the original U-net model in the previous research as the baseline model and compare the model performance with the model trained from scratch. The results demonstrate that even though the transfer learning model leads to better performance and less computation power, it has limitations that need to be considered. 


## How to use the notebooks
1. Download the data from the link below  
-- will be provided
2. Run preprocessing.ipynb notbook for preprocessing part of the experiment
3. Run experiment_1.ipynb notebook for experiment 1 the results should be in training_results/experiment_1/ folder
4. Run experiment_2.ipynb notebook for experiment 2 the results should be in training_results/experiment_2/ folder
5. Run experiment_3.ipynb notebook for experiment 3 the results should be in training_results/experiment_3/ folder
6. Run evaluation.ipynb notebook to evaluate the prediction results
7. The conclusion.ipynb is to show the conclusion of the research

