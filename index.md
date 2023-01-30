# MSC-Final-Project- Spatial Transcriptomic Imputation

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