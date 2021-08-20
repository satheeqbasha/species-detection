# species-detection
# Implementation of EfficientNet and YOLO models on fish species dataset

Name: Atheeq Basha Syed
, Department of Computer Science, University of Exeter 

Project Name: Underwater Video Analysis for Fish Biodiversity Monitoring

All reqiurements are available in the reqirements.txt file

The methods which would be necessary in implementing models are present in prog_help.py file.

Instructions to follow before running the python notebook:
1. Connect to a GPU-enabled network or Computer to experience the full potential of the experiment.
2. The notebook was trained on the Nvidia RTX 2060 GPU and local CPU which supports the CUDA 10.1 and CuDNN 7.6 version for execution. 
3. To execute this notebook in one's local system, please install proper CUDA and CuDNN versions to correctly enable the GPU feature based on your local Graphic Card for smooth execution.
4. To execute in Google Colab, load the notebook in the colab environment with GPU enabled and execute the code blocks below.    

Instructions to run the python notebook:
1. Open the python notebook (fish_detection_notebook.ipynb) on a Python Server (preferably Google Colab as it provides better GPU support or Jupyter Notebook)
2. Follow the instructions on the Python Notebook to execute the experiment.
3. Either go to Runtime > Run all command (Ctrl+F9) or execute the code based on Code Blocks. There are five code blocks in total. 
4. Block 1 contains the code block execution that downloads the required data from a zip file present in a DropBox folder and downloads and imports all the necessary libraries.
5. Block 2 contains the code which executes the pre-trained YOLO and EfficientNet models on the Open video and Fish4Knowledge dataset video.
6. One can access the detections on the video by playing the videos present in the link './videos/Fish_in_Sunlight_Videvo_yolo.mp4' and './videos/video_siteNPP-3_camera1_201104221410_yolo.mp4'
7. Block 3 contains the code which holds all the required analysis on the training and testing data
8. Block 4 contains the code which executes the YOLO model on the Images from the sample video.
9. Block 5 contains the code which downloads the training data and executes the training on new EfficientNet and YOLO models.
10. Usually, block 5 takes 5 to 6 hours to complete its execution. So, the execution of the notebook is interrupted for Block 5. If one wishes to execute block 5, just re-run the Block after interrupt.
11. Once Block 5 is executed, run blocks 2 to 4 to observe the detections from the newly trained model.

