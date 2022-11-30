# ml-depression-classification
Given a comment, determine whether that comment may indicate depressive tendencies in the author. Not meant for production use, just a toy project.


## Project organization

The `data` folder (ignored on  github) holds data from kaggle. Download the 
[dataset](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned?resource=download)
from the original publisher. 

The `exploratory_analysis` holds scripts to perform EDA and text cleaning.

The `model_building` directory holds code to train a classifier on the data.

The `depression_classification` directory holds the python package. 
This package can be installed and imported with
```
pip install -e .
python
>>> import depression_classification as dc
```
