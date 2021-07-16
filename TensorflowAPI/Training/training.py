"""
Training is done using a MobileNet model & Tensorflow Object Detection Model
MobileNet - architecture
"""

import os
import wget
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import TensorflowAPI.collectAndLabel.image_labelling as il

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
    # different checkpoints for the pre trained model
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

# Download TensorFlow2 Object Detection ReusableAPI:

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


# Install TF ReusableAPI
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
il.execute("pip install pytz")

# We download the TF model for:
# Leverage Architecture
# Make use of Transfer Learning technique

# This model is already trained to detect different objects, we are going to leverage this and make it
# detect custom objects of ours

if os.name =='posix':
    il.execute("wget " + PRETRAINED_MODEL_URL)
    il.execute("mv " + PRETRAINED_MODEL_NAME+'.tar.gz' + paths['PRETRAINED_MODEL_PATH'])
    il.execute("cd " + paths['PRETRAINED_MODEL_PATH'] +" && tar -zxvf " + PRETRAINED_MODEL_NAME+'.tar.gz')

if os.name == 'nt':
    wget.download(PRETRAINED_MODEL_URL)
    il.execute("move " + PRETRAINED_MODEL_NAME+'.tar.gz' + paths['PRETRAINED_MODEL_PATH'])
    il.execute("cd " + paths['PRETRAINED_MODEL_PATH'] + " && tar -zxvf " + PRETRAINED_MODEL_NAME+'.tar.gz')

# Creating the label map
# Creates a dictionary, in form of JSON, with 4 items: which include name & id
# This is saved as a text file in LABELMAP path (Tensorflow-> workspace-> annotation)

labels = [{'name':'ThumbsUp', 'id':1}, {'name':'ThumbsDown', 'id':2}, {'name':'ThankYou', 'id':3}, {'name':'LiveLong', 'id':4}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


# ---------------------------------- Creating TF (model) RECORDS ------------------------------------------------------
"""

> Using TF Records, we train our model
> Tf records are a binary file format for storing data
> They speed up training for your custom Object Detection model
> They convert the annotations (Xml) and photos (Jpeg) into a file format that they can use

"""

if not os.path.exists(files['TF_RECORD_SCRIPT']):
    il.execute("git clone https://github.com/nicknochnack/GenerateTFRecord " + paths['SCRIPTS_PATH'])

il.execute("python " + files['TF_RECORD_SCRIPT']+" -x " + os.path.join(paths['IMAGE_PATH'], 'train') + " -l " +
           files['LABELMAP']+ " -o " + os.path.join(paths['ANNOTATION_PATH'], 'train.record'))
il.execute("python " + files['TF_RECORD_SCRIPT']+" -x " + os.path.join(paths['IMAGE_PATH'], 'test') + " -l " +
           files['LABELMAP']+ " -o " + os.path.join(paths['ANNOTATION_PATH'], 'test.record'))

"""
Successfully created the TFRecord file: Tensorflow\workspace\annotations\train.record
Successfully created the TFRecord file: Tensorflow\workspace\annotations\test.record
"""

# ------------------ Copying the (default) Model Configuration to our local training directory -------------------------
# This file is called pipeline.xml -> which is an XML file containing information about features/paths of our model

if os.name =='posix':
    il.execute("cp " +os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config') + " " +
               os.path.join(paths['CHECKPOINT_PATH']))
if os.name == 'nt':
    il.execute("copy " + os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config') + " " +
               os.path.join(paths['CHECKPOINT_PATH']))
"""
        1 file(s) copied.
"""


# --------------------- Config the pipeline.xml file ------------------------

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
"""
config will look like:

{'model': ssd {
   num_classes: 4
   image_resizer {
     fixed_shape_resizer {
       height: 320
       width: 320
     }
   }
   feature_extractor {
     type: "ssd_mobilenet_v2_fpn_keras"
     depth_multiplier: 1.0
     min_depth: 16
     conv_hyperparams {
       regularizer { ......
       
"""
#
#
#
#
#
# from object_detection.protos import pipeline_pb2


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Sets up file paths & configuration values
pipeline_config.model.ssd.num_classes = len(labels)
# change the number of classes (type of objects)
pipeline_config.train_config.batch_size = 4
# set path for checkpoint directory
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

# Saves the configuration
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)


# ---------------------------------------------------- Training -------------------------------------------------------
# Runs: 1) Tensorflow\models\research\object_detection\model_main_tf2.py
#       2) (our pipeline configuration at:) Tensorflow\workspace\models\my_ssd_mobnet
#       3) Set parameters (2000 steps prototyping)


# !!! The environemnt has to be activated (tfod - shown at the beginning), otherwise the ReusableAPI is not activated

TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT,
          paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])

print(command)
"""
comamand:

python Tensorflow\models\research\object_detection\model_main_tf2.py \
--model_dir=Tensorflow\workspace\models\my_ssd_mobnet \
--pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config \
--checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet\

"""

il.execute("pip uninstall pycocotools -y")
il.execute("pip install pycocotools")
il.execute("pip install gin-config==0.1.1")
il.execute("pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1")
il.execute(command)

print("Please wait until all 2000 steps are finished")
print("You should now see the output of the model (training, ms ...)")
print("You should also have checkpoint files generated: ckpt[...].index")


# ----------------------------- Evaluate the model -----------------------------------------
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
print(command)

# ------------------------------ See graphs -------------------------------------------------

il.execute("cd Tensorflow\\workspace\\models\\my_ssd__mobnet\\train")
il.execute("tensorboard --logdir=.")
print(" > Please go to the link mentioned in the cmd [should be localhost]")



def load_checkpoint(latest_checkpoint):

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-'+latest_checkpoint)).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

load_checkpoint(3)


"""
This will show the image + the box around it (and the accuracy)
"""
def detect_image(img_name):
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', img_name)

    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False)

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()


def launch():
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


launch()