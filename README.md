Overview
This project is based on the adaptive ellipse DBSCAN algorithm for ICESat-2 bathymetric point extraction and then inversion of bathymetric maps based on ICESat-2 extracted bathymetric points combined with Gaofen images. 
The project uses Matlab and Python.
Scripts
Training steps
•	Configure the ICESat-2 datasets in pycode/step_1_savedata.py for training
•	Run pycode/online_dbscan.py for ICESat-2 bathymetric point extraction
•	Run matlabcode/step1_outlier.m, step2_transfer.m, step3_grid.m, step4_transfer_bathy.m for ICESat-2 bathymetric points alignment with Gaofen images before training.
•	The results are presented in controlpoits/
•	Run matlabcode/ save_rnn.m for the recurrent neural network training.

Validation steps
•	Load the test image in image/.
•	Run matlabcode/ dr_huaguangjiao_gf6.m for validation
•	The results are presented in codes/image/res/

