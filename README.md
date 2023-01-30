# Spatial Transcriptomic Imputation Project
## Overview
Analysis of gene expression profiles in biological samples is a powerful tool used to study various biological systems and phenomena. Traditional assays measure bulk gene expression levels of relatively large samples (i.e. containing millions of cells) yielding robust measurements but hiding cell to cell variability. In the last decade, high throughput single cell RNA-Seq (scRNA-Seq) technologies were developed to capture this variability for thousands of cells simultaneously allowing for in-depth analysis of biological tissues. However, even with single cell resolution, the spatial information over the measured tissue is lost with scRNA-Seq. Recently, new technologies measure gene expression profiles of biological tissues while maintaining spatial information. These Spatial Transcriptomics (ST) technologies allow for studying complex tissues where direct interactions between different cell types affect the biological system. For example, when studying cancerous samples, the effect of the tumor microenvironment is directly associated with the disease stage, treatment decisions and survival rate.

![Introduction1](https://user-images.githubusercontent.com/59770634/215389841-49b45800-d811-49bf-8e5c-652357f2820c.PNG)

## Goal
The given generated expression data from visium is very sparse, capturing only a partial view of the complete gene expression profile in each spot. This is called the depth limitation.
In this project we will explore methods to perform data imputation on ST datasets in order to overcome the depth limitation. We will develop models and techniques for using external inputs (bulk RNA measurements, scRNA-Seq, pathology images) and the spatial information to enrich the ST information.
We will embed the gene expression data into a latent space using matrix factorization technique, and will use the spatial information to help us enrich the data in order to to better reconstruct the gene expression information of the spots. Also, we will use information from the image of the tissue to produce more helpful information from it and enrich the data.
![0s histogram](https://user-images.githubusercontent.com/59770634/215392106-eeb7ad88-a930-44ec-a73b-4b96e9b4df80.PNG)

## Solution
In this project we used DL techniques to impute the missing values. We embed the gene expression into a latent space and reconstruct it back to its original shape with the new predicted values. We use vanilla Neural Matrix Factorization, NMF, method as a baseline model and compaison criteria to our developed methods.
In addition to the basic NMF, we use spatial information around each spot in order to "smooth" the imputed predictions. The spatial information of the neighbors of each spot is retrived from the original matrix. Also, in order to use the most relevant data from the neighbors we used the information found in the image itself.
### trials
- In one trial, we tried to smooth each spot's data using only the spots located in the same region of it using the image. This method should be more explored in future work.
- In other, more succesful trial, we filtered the irrelevant data using different tile sizes around each spot, and used the RGB values in each one of the tiles to compared them. Only spots with tiles with RGB values close to the center spot were used during the smoothing procedure.

## Results
Comparison between each one of the techniques implemented in the project is presented in the below graph.
![results bar graph](https://user-images.githubusercontent.com/59770634/215393172-3612fe86-e81d-480d-8cd0-e84e2ec0f076.PNG)

## How to Run the Code
Please clone the current repository to your computer.
To run it through a container follow the below instructions:
- Open terminal on the directory path
- Build and start the container: `docker-compose up -d`
- Download the 'data/' directory from the google drive directory-
'https://drive.google.com/drive/folders/11OodT6oUUyOpqNdZiJQWsAhWbFOE4i0Z?hl=he'
- Run the code in one of the following ways-
  1. - Open your internet browser (Chrome / Safari / ...)
     - Enter the Jupyter notebook URL: `http://localhost:8888`
     - Open the 'main notebook.ipynb' and execute the code blocks one after another.
  2. Open the 'train.py' file in an IDE (VScode for eample) and execute- Make sure that the function 'train_data_for_imputation' is being called from the 'main' function.


