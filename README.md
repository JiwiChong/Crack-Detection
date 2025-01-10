# Crack-Detection

Crack Detection is a critical task in the field of Civil Engineering. Presence of cracks 
is a critical sign of deterioration of structures and other types of infrastructures. In fact, 
failure to find structural cracks results in 46% of structural weakness and hance, vast economic
expenses. 

In this work, effective Convolutional Neural Networks models proposed by two top-quality journals 
were implemented. The first one is CF0 and the othe one is Multi-Scale CNN Network. The first model was proposed from
the journal "Expert System with Applications" (Impact Factor 7.5). Multi-Scale CNN Network is the model I developed and propose
for the task in this repository. The task is to classify the images
as "Crack" or "Non-Crack". The dataset involved in this work is SDNET2018, a large dataset of imbalanced
images. The task with this dataset is highly challenging due to such imbalance as well  as the presence
of images that contains parts that resemble crack but actually aren't crack.

**_CF0_** <br />
**_Multi-Scale CNN Network_** 

With Multi-Scale CNN Network, which learns by involving parallel channels and high pass kernels 
at early stage, the crack detection Accuracy score was improved from **64%** to **84%**. 

<div align="center">
<img src="https://github.com/user-attachments/assets/89fd35ae-3aa8-4449-aa29-77de9ffa8deb" width=80% height=80%>
</div><br />

Multi-Scale CNN Network was then applied for a Transfer Learning process to be finetuned for a new task, which is
the image classification of Metal images, whether they have damage or no damage. If they have damage,
they would be classified based on the type of damage. 

### Commands
**_Train_** <br />
```python
python main_dev.py --run_num (# of run) --model_name (Name of the Model) --epochs (# of epochs to train) --rgb (Whether image is in RGB or not) <br />
```

**_Evaluation_** <br />
```python
python eval.py --run_num (# of run) --model_name (Name of the Model) --data (Name of the data) --image_form (Whether image is in RGB or not) --num_workers (# of CPU workers to train) <br />
```
