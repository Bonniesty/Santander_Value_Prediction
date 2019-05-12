# Santander Value Prediction

## Brief introduction
Our submitted code combines Feature Learning (including Principle Components Analysis and k-means..) and LightGBM algorithm to train the model and predict the target data. 

Since our dataset is too big to upload, please download our train data and test data [HERE](https://drive.google.com/drive/folders/1xS0IMsJD8dm8lKYizYs1pVxsSMtC1SrX?usp=sharing). Download our repo and put raw data file in ```data``` folder.

## How to run the code
To run our model and see the result, please run the following command. Make sure you have defined the directory of the dataset in the code. You are also required to install some libraries. 

```python 
python3 model.py
```
## What you will see
Training result, Graphic Analysis can be found in the console. The code will generate a submission file for prediction result.

To see the plot of PCA and variance, you can open the ```Santander_Value_Prediction_Plot.ipynb``` file in notebook and run all the cells.

