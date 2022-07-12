#System Requirements:
- anaconda3
- Opencv3.5, Numpy, Matplotlib
- Pytorch3, Python 3.6.9

########################################
#### Evaluation of Hierarchical approach
########################################
   

#Final Evaluation of GAMa-Net model on updated/new gallery: 

- Run the following script for evaluation
	python ./GAMa_Net/evaluationC_pred_Laerial_test_top1per.py

-- Please note that the first time evaluation will take longer (few hours) to create a dictionary; after that evaluation will be faster
	
More details:

##################################################
###Training and evaluation of both steps
###################################################

#Training GAMa-Net model:
- Run the script to train
	python ./GAMa_Net/main.py 
	(Update '--data_folder' in main.py)
	
-Please update the dataset path and change the batch size as per the computing power. See file job.sh for prior training details.

#Evaluating GAMa-Net model:
- Run the following scripts, in the given order, for evaluation
	python ./GAMa_Net/process_videos_C.py # saves video embedding
	python ./GAMa_Net/process_gallery_C.py # saves gallery embedding
	python ./GAMa_Net/evaluationC_pred_test.py # evaluates 

# Allow a dictionary to load at the start. This may take few minutes.
		

#########################

#Training screening model using ground truth:
-Run the script:  
	python ./screening_net/main.py

#Finetuning screening model using prediction from GAMa-Net on training-list:

-First update the list in dataloader and evaluation file
-Run the following script to save a dictionary for training screening network using GAMa-Net evaluation on training-list
 - python ./GAMa_Net/process_videos_C.py # saves video embedding
 - python ./GAMa_Net/process_gallery_C.py # saves gallery embedding
 - python ./GAMa_Net/evaluationC.py # evaluates and saves dictionary 
	
-Run the script:  
	python ./screening_net/main_finetuning.py

#Evaluating  screening model:
- Run the following scripts for evaluation
	python ./screening_net/process_video_C.py # saves aerial sequence embedding
	python ./screening_net/process_gallery_C.py # saves larger aerial region embedding
	python ./screening_net/evaluationC.py # evaluates and saves a dictionary for reduced gallery for GAMa-Net
	
########################################

#Hierarchical approach

#Final Evaluation using  GAMa-Net model on new gallery: 

- Run the following script for evaluation
	python ./GAMa_Net/evaluationC_pred_Laerial_test_top1per.py
	
######################################################
########################################################