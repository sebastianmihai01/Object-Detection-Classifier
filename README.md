# Documentation TFOD (Tensorflow Object Detection)
## Notes for creators:
 - Done with TFOD API
 - Leveraged Camera (Labelling and identifying the objects)
 - Training (Static & Dynamic-real time- Detection) is done via labelling
 - Data: Image ... Answers: Annotations/Labels ... Trained the ML model with these
 - Freeze (save&load) model
 - Export and Deploy (Reusable models) / Convert into 'tf' format
 - Perform tuning (solving wrongfully detected objects)
 - Training on Google Cloud (saves RAM)
 - TBT: Rasperry PI integration (coming in September)
 - CudaNN (and CUDA) gives GPU acceleration when doing model training (slower on CPU)
 - Create enviroment (isolate libs and dependencies) via 'python -m venv tfod' (create a notebook in Jupyter - run with 'jupyter notebook' cmd)

## Installation
 - Make sure the GPU, VS version, TF version, cuDNN & CUDA versions are matching\
 as in: https://www.tensorflow.org/install/source_windows
 - python -m pip install --upgrade pip
 - pip install ipykernel (associate the environement with our jupyter notebook)
 - python -m ipykernel install --user --name=tfodj (install our virtual environment into jupyter)

## Algorithms & Approaches:
- MobileNet Convolutional NN model used\
(read more on: https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470)
- Depthwise Separable Convolutional Layering

#
 <img  width="80%" height="80%" align = "center" src ="https://miro.medium.com/max/1384/1*7R068tzqqK-1edu4hbAVZQ.png">
 (source: https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470)
