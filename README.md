# Project Report

**Project Goal:**  
The aim of this project is to develop an application that predicts diseases based on given symptoms.

## Project Steps

### Dataset Exploration
During the initial examination of the dataset, it was observed that diseases with the same name sometimes had their symptoms listed in different columns. This inconsistency could lead to incorrect results during model training. Initially, I considered merging all symptoms of the same disease into a single row. However, this approach resulted in overlapping symptoms in the same columns, which significantly reduced the dataset size (by approximately 99%).

### Data Preprocessing
To address this issue, I extracted all unique symptom names into a separate list and created a new, empty dataset containing only the disease names. For each disease, the symptoms from the original dataset were matched with the new symptom columns, and the corresponding cells were marked with a value of 1.  
Subsequently, this binary dataset was multiplied with another dataset containing symptom weights, replacing the 1’s with their respective weight values.

### Model Development and Training
A neural network model was implemented. The number of neurons in the input layer was set equal to the number of input features. Hidden layers used the **ReLU** activation function, while the output layer used **Softmax** for multi-class classification. To prevent overfitting, techniques such as **regularization, dropout, and early stopping** were applied.  
After training, the model was evaluated with appropriate tests and then saved for deployment.

### Web Application Integration
A web interface was developed, and the trained model was integrated into it using **Flask**. When a user selects symptoms on the website, the system applies the appropriate weightings, passes them through the model, and generates a disease prediction. In addition, two separate datasets — Symptom Description and Symptom Precaution — were used to display disease explanations and recommended precautions to the user.
