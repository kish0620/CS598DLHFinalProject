# CS598DLHFinalProject
Step 1: Install Mimic III Clinical Database
Go to this link: https://physionet.org/content/mimiciii/1.4/
If you don't have the required CITI Data or Specimens Only Research training, complete the necessary modules to gain access
To install the database, go to the Access the files sub section of the Files section and choose either of the four options
Unzip the folder once it has been downloaded

Step 2: Clone repo and move dataset to project folder
Run git clone https://github.com/kish0620/CS598DLHFinalProject.git in your desired directly to clone the project over
cd into the newly added CS598DLHFinalProject directory and move the dataset you installed in step 1 into this directory

Step 3: Install dependencies and run file
Install the required packages with the requirements.txt file
In reproduction.py, choose between the RNN or Transformer with line 215 or 217 respectively.
Run python reproduction.py to preprocess, train, and evaluate the model
