# EEG-Based-Objective-Measurement-of-Temporal-Profiles-of-Emotion-Intensity
This is the PyTorch implementation of the training process in our reserach:
"EEG-Based Objective Measurement of Temporal Profiles of Emotion Intensity"

In this work, we are investigating emotion intensity profile based on EEG signals.
# install
> pip3 install -r requirements.txt
```
argparse
torch==1.8.0
h5py==3.2.0
numpy==1.20.1
scikit-learn==0.24.1
scipy==1.6.2
```
# Run the code
> python main.py

# Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --data-path 	            |	           |DEAP raw file path
| --start-subjects          | 0           |the starting subject number (from 0 to 31)
| --subjects	              |	32	            |the number of training subjects
| --label-type  		        | 'A'	           | emotion label ('A' : arousal 'V':valence)
| --round 		              | 5             | baseline training repeated round
| --weight_decays 	        | 1e-2           | 
| --save-path	              | './save'       | the trained model saving file path
| -testlist                 | False           | if test list existed ( True: exist; False: split train/test trials)
| --testlist_file           | './save/arousal/'  | test list file path
| --max-epoch               | 150            | training epochs for baseline and new model training
| --ft-epoch			             | 50 	           | training epochs for fine tune training
| --threshold			          | 20         | the percentage of selecting sub-dataset size 
| -retrain-type			             | 'ft'     	     | retrain method ('ft': fine tune, 'nm': new model)
| --selection		    | 'time'     	     | selection method ('time': time interval method; 'score': score method)
| --batch-size		      | 128     	   | batch size for training dataset

