# EEG-Based-Objective-Measurement-of-Temporal-Profiles-of-Emotion-Intensity
Repetition code of the model for the paper "EEG-Based Objective Measurement of Temporal Profiles of Emotion Intensity" in pytorch


In this work, we are investigating emotion intensity profile based on EEG signals.
# Requirements
> install
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
| --weight_decays 	        | 1e-2           | optimizer weight decays
| --save-path	              | './save'       | the trained model saving file path
| -testlist                 | False           | if test list existed ( True: exist; False: split train/test trials)
| --testlist_file           | './save/arousal/'  | test list file path
| --max-epoch               | 150            | training epochs for baseline and new model training
| --ft-epoch			             | 50 	           | training epochs for fine tune training
| --threshold			          | 20         | the percentage of selecting sub-dataset size 
| -retrain-type			             | 'ft'     	     | retrain method ('ft': fine tune, 'nm': new model)
| --selection		    | 'time'     	     | selection method ('time': time interval method; 'score': score method)
| --batch-size		      | 128     	   | batch size for training dataset

# github format
```
.
├── github                   # Score code files (alternatively `dist`)
│   ├── main.py              # Main file to execute research 
│   ├── SCCNet.py            # Model architecture
│   ├── channel_baseline.txt # Channel order for prepare data
│   ├── func.py              # Functional tool use in research
│   ├── generate_profile.py  # Code for generating emotion intensity value for trials
│   ├── preprocess.py        # Prepare raw data in DEAP dataset
│   ├── requirements.txt     # Requirement detail
│   ├── score.py             # Score selection and retraining process
│   ├── time_interval.py     # Time-interval selection and retraining process
│   └── training.py          # Baseline model training process
└── README.md

```

