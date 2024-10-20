# Crack-Detection

Crack Detection is a critical task in the field of Civil Engineering. Presence of cracks 
is a critical sign of deterioration of structures and other types of infrastructures. In fact, 
failure to find structural cracks results in 46% of structural weakness and hance, vast economic
expenses. 

In this work, effective Convolutional Neural Networks models proposed by two top-quality journals 
were implemented. The first one is CF0 and the othe one is MultiScaleNet. Both models were proposed from
the journal "Expert System with Applications" (Impact Factor 7.5). The task is to classify the images
as "Crack" or "Non-Crack". The dataset involved in this work is SDNET2018, a large dataset of imbalanced
images. The task with this dataset is highly challenging due to such imbalance as well  as the presence
of images that contains parts that resemble crack but actually aren't crack.

**_CFO_** [Paper](https://www.sciencedirect.com/science/article/pii/S0957417423009491)<br />
**_MultiScaleNet_** [Paper](https://www.sciencedirect.com/science/article/pii/S0957417424005244)

With MultiScaleNet, which learns by involving parallel channels and high pass kernels at early stage, 
the crack detection Accuracy score was improved from **64%** to **84%**. 

![test_image](https://github.com/user-attachments/assets/89fd35ae-3aa8-4449-aa29-77de9ffa8deb width=200 height=250)

MultiScaleNet was then applied for a Transfer Learning process to be finetuned for a new task, which is
the image classification of Metal images, whether they have damage or no damage. If they have damage,
they would be classified based on the type of damage. 
