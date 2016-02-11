# faceClassifier
Face Classifier using a cascade of CNNs

Files structure:

          ---faceClassifier
              ---demo                 simple demo for illustrating basic idea
              ---preprocess           convert weights from lua file to binary file
              ---src                  main program
              ---test                 test images
              ---weights              weights of each layer

###Run **main program**:

Step 1:
Open file "..../faceClassifier/src/model.c";

Step 2:
Modify the **FILE_PATH[]** to the path which your folder ***..../faceClassifier*** is on;

Step 3:
(optional) Change the test image path **TEST_IMAGE[]** (relative path);

Step 4:
If this is the first time you try to run the program, delete all files in folder ***..../faceClassifier/src/build/***;

Step 5:

Open **terminal**, 

**cd ..../faceClassifier/src/build/**,

**cmake ..**,

**make**,

**./run**



###Run **demo**:

Step 1:
Open file "..../faceClassifier/demo/model.c";

Step 2:
Modify the **FILE_PATH[]** to the path which your folder ***..../faceClassifier*** is on;

Step 3:
(optional) Change the test image path **TEST_IMAGE[]** (relative path);

Step 4:
If this is the first time you try to run the program, delete all files in folder ***..../faceClassifier/demo/build/***;

Step 5:

Open **terminal**, 

**cd ..../faceClassifier/demo/build/**,

**cmake ..**,

**make**,

**./run**
