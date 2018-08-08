# Gender-Identification-Using-Fishaerface

## Getting Started :-

This Project aims to find gender of a person using fisherface by directly extracting images.

  - Gender identification using fisherface in Python.
  - Data storage hierarchy where data files are in src and images are in male and female folder in raw_gender.
  ```
     ├── data
        ├── raw_gender
          ├── male
          ├── female
     └── src
   ```
     
## About Model :-

The included models are essential for the program to detect faces, emotions, and genders.

### HaarCascade :

These models are provided by OpenCV and allows the program to detect human faces. After some manual and automated testings, I decided to use the first alternate version. If for some reason you want to change the way this program detect human faces, open face_detection.py, search the following line: faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml') and change the model path to the desired one.

Gender Classifier
These models are created with train_gender_classifier.py. There are 3 versions: normal, KDEF, and IMDB. The normal version is trained with both KDEF and IMDB datasets. While KDEF and IMDB is trained with just KDEF or IMDB respectively.

Due to memory limitation only a handful of photos (2000+) from IMDB is used in building the normal and IMDB version. The best result is indeed achieved using the normal version which combined both KDEF and IMDB.

To switch versions, open facifier.py and search the following line: fisher_face_gender.read('models/gender_classifier_model.xml') and change the model path to the desired one.

Windows/Linux
It turns out that a model trained using Windows can only work in Windows and that also applies to Linux. A new Windows-friendly model has been added to the model directory.

For anyone using Windows, go to line 69-73 in src/facifier.py.
<pre>
fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
fisher_face_emotion.read('models/emotion_classifier_model.xml')

fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
fisher_face_gender.read('models/gender_classifier_model.xml')
</pre>
Change them into:
<pre>
fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
fisher_face_emotion.read('models/emotion_classifier_model_windows.xml')

fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
fisher_face_gender.read('models/gender_classifier_model_windows.xml')
</pre>
The application should work properly in Windows with the new models.

### Why Fisherface?

Fisherface is not the only available recognizer in OpenCV, so why specifically choose it over the others?

For starters, it is better than using Eigenface recognizer. Eigenfaces are the eigenvectors associated to the largest . It uses integer value in its prediction. This value is the corresponding eigenvalue of the eigenvector.

Meanwhile, Fisherface uses Linear Derived Analysis (LDA) to determine the vector representation. It produces float value in the prediction. This also means that the result is better compared to Eigenface.
     
## Prerequisties :-

  - A 64-bit operating system(like Ubuntu, Windows or mac.......)
  - Python 3.6
    - For Windows you can install Anaconda or python IDE.
    - For ubuntu you can either work on conda or your terminal.
  - OpenCV
    - For windows you can install [openCV in Anaconda.](https://www.geeksforgeeks.org/set-opencv-anaconda-environment/)
    - For Ubuntu you can install [openCV using terminal.](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html)
  
## Running :-
  - On Ubuntu :-
  
    - $ python3.6 &lt;Program file&gt;
  
  - On Ide :-
  
  - Open any Python3 Ide and Run the program.

    - First set path of compiler.
    - Type Python3.6 &lt;Program file&gt;
    
## Running Program Sequence :-
  - To run program run program files in following sequence
    <pre> gender_data_prep.py  -&gt;  train_gender_classifier.py </pre>
  
