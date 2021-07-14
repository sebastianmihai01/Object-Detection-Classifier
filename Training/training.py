"""
Training is done using a MobileNet model & Tensorflow Object Detection Model
MobileNet - architecture
"""

import os
import wget
import collectAndLabel.image_labelling as il
import object_detection  # Object detection API from tensorflow 2

# Our folder
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

# Model name from the TF repository
# Taken from the download link of the model - it is written at the end of the link (see below)
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

# Download link of the model
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'


TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

# Setting the paths
# Just type: paths['name'] => output will be the path
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }


# Paths for files
# type: files['name']

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Set up folder structure
# Create folders for all the paths (mentioned above)

for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            il.execute("mkdir -p" + path)
        if os.name == 'nt':
            il.execute("mkdir " + path)

# Download TensorFlow2 Object Detection API:

# https://www.tensorflow.org/install/source_windows
if os.name=='nt':
    il.execute("pip install wget")


if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    il.execute("git clone https://github.com/tensorflow/models " + paths['APIMODEL_PATH'])


# Install Tensorflow Object Detection
if os.name=='posix':
    il.execute("apt-get install protobuf-compiler")
    il.execute("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py. && python -m pip install .")

if os.name=='nt':
    # install protobuf (Google's Protocol Buffers)
    url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url)
    il.execute("move protoc-3.15.6-win64.zip " + paths['PROTOC_PATH'])
    il.execute("cd " + paths['PROTOC_PATH'] + " && tar -xf protoc-3.15.6-win64.zip")
    os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))


# Install TF API
il.execute("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install")
il.execute("cd Tensorflow/models/research/slim && pip install -e .")


# Verify if the installation is correct
VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
il.execute("python " + VERIFICATION_SCRIPT)

# Install tf
il.execute("pip install tensorflow --upgrade")

# Install tf on GPU
il.execute("pip install tensorflow-gpu --upgrade")

# Update and/or install needed packages
il.execute("pip uninstall protobuf matplotlib -y")
il.execute("pip install protobuf matplotlib==3.2")
il.execute("pip install PIL")
il.execute("pip install pyyaml")
