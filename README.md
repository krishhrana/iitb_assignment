# Vehicle classification
The final.py file contains the source code for vehicle classification model and the testfinal.py file loads the CNN model for implementing the model on new images.
The CNN model is based on tensorflows implentation of keras library.The model classifies the vehicles into three categories (Cars, Autos, Motorcycles)
# Python libraries used
1.numpy   
2.os    
3.tensorflow   
4.opencv as cv2     
5.tensorflow.keras (tensorfow implementation of keras).     


# Code and Files
**1.My project includes following files**
1. final.py     
2. testfinal.py     
3. numberplate.py    


**2.Environment and system configurations**
I have used Sublime text as my text editor. Python version is 3.7.3 OS:macos 10.14.5




**3.How to run the code **
Save the Dataset on your computer and set the DATA_DIR as the filepath
```python
DATA_DIR = "filepath"
```
Dataset link-https://www.kaggle.com/krishrana/vehicle-dataset      
To use the saved model, run the testfinal.py file and enter the path to the image in the prepare()
```python
pred = model.predict([prepare("filepath")])
```    
The training accuracy achieved is 99.02%     
The test accuracy achieved is 86.86%

# Number Plate detector

