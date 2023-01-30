"# kdai_repo" 
# Train an Object Detection Classifier Using TensorFlow (GPU) on Windows 10

## Steps
### 1. Install Anaconda, CUDA, cuDNN and Tensorflow-GPU
  1.1. ติดตั้ง Anaconda ตาม version python ที่ต้องการ ณ ที่นี้แนะนำ python 3.7
  
  1.2. ติดตั้ง CUDA Toolkit v10.0
  
  1.3. ตั้งตั้ง CuDNN 7.6.5
  
  #### install cuDNN
  โหลด cuDNN แตกไฟล์วางไว้ที่ 
  
  ```<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\```
  
  #### Environment Setup
    Edit the system environment variables
```
<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
    
<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp

<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64

<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.0\cuda\bin

```

*Update your GPU drivers

####Create a new Conda virtual environment

```

conda create -n tensorflow_gpu pip python=3.7

```
```
activate tensorflow_gpu
```

  ### Install TensorFlow GPU for Python
 ```
 pip install --upgrade tensorflow-gpu==1.13
 ```

  ### Test your Installation

```
python
```

```python
import tensorflow as tf
```

```python
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
```
หากสามารถใช้งานได้จะแสดงบางอย่างที่หน้าตาประมาณ

Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6358 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)


```python
print(sess.run(hello))
```

  #### ติดตั้ง packages
```
(tensorflow_gpu) C:\> conda install -c anaconda protobuf
(tensorflow_gpu) C:\> pip install pillow
(tensorflow_gpu) C:\> pip install lxml
(tensorflow_gpu) C:\> pip install Cython
(tensorflow_gpu) C:\> pip install contextlib2
(tensorflow_gpu) C:\> pip install jupyter
(tensorflow_gpu) C:\> pip install matplotlib
(tensorflow_gpu) C:\> pip install pandas
(tensorflow_gpu) C:\> pip install opencv-python
```
#### Configure PYTHONPATH environment variable
```
(tensorflow_gpu) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```

  #### Compile Protobufs and run setup.py

 From within TensorFlow/models/research/
```
(tensorflow_gpu) protoc object_detection/protos/*.proto --python_out=.
```
```
(tensorflow_gpu) C:\tensorflow1\models\research> python setup.py build
(tensorflow_gpu) C:\tensorflow1\models\research> python setup.py install
```

  #### Test TensorFlow setup to verify it works
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
ถ้าสามารถรันได้ถึงดีเทคหมาน้อย แปลว่าใช้งานได้

### 2. Data set

สร้าง folder workspace ด้านในมี folder training_demo
path จะประมาณนี้ (ไม่ต้องทำตามก็ได้)
```
TensorFlow
├─ models
│   ├─ official
│   ├─ research
│   ├─ samples
│   └─ tutorials
└─ workspace
    └─ training_demo
        ├─ annotations
        ├─ images
        │   ├─ test
        │   └─ train
        ├─ pre-trained-model
        ├─ training
        └─ README.md
```
  annotations ไว้เก็บ .csv and .record or .tfrecord
  training ไวเก็บ .config and .pbtxt
  pre-trained-model เลือก model จาก https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
  โหลดและแตกไฟล์เก็บไว้ (ในที่นี้แนะนำ Faster RCNN)
  
  
  #### Annotating images
  สร้าง lebel ให้ dataset สามารถใช้ได้หลายโปรแกรม ในที่นี้ยกตัวอย่าง https://www.makesense.ai/
  
  #### Partitioning the images
  แบ่ง dataset for train and test
  
  ```
  python partition_dataser.py -x -i training_demo\images -r 0.1
  ```
  
  #### Creating Label Map
  สร้างไฟล์ label_map.pbtxt สำหรับกำหนด id label ที่เราต้องการ detection
  ตัวอย่างภายในไฟล์
  
  ```
  item {
    id: 1
    name: 'cat'
}

item {
    id: 2
    name: 'dog'
}
```

or

```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'nine':
        return 1
    elif row_label == 'ten':
        return 2
    elif row_label == 'jack':
        return 3
    elif row_label == 'queen':
        return 4
    elif row_label == 'king':
        return 5
    elif row_label == 'ace':
        return 6
    else:
        None
```

  #### Creating csv and tfrecord
  
  ```
  python xml_to_csv.py
  ```
  (ไปแก้ path ใน xml_to_csv.py ก่อน)
  ```
  # Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv
--img_path=<PATH_TO_IMAGES_FOLDER>/train  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv
--img_path=<PATH_TO_IMAGES_FOLDER>/test
--output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record

# For example
# python generate_tfrecord.py --label=ship --csv_input=C:\Users\sglvladi\Documents\TensorFlow\workspace\training_demo\annotations\train_labels.csv --output_path=C:\Users\sglvladi\Documents\TensorFlow\workspace\training_demo\annotations\train.record --img_path=C:\Users\sglvladi\Documents\TensorFlow\workspace\training_demo\images\train
# python generate_tfrecord.py --label=ship --csv_input=C:\Users\sglvladi\Documents\TensorFlow\workspace\training_demo\annotations\test_labels.csv --output_path=C:\Users\sglvladi\Documents\TensorFlow\workspace\training_demo\annotations\test.record --img_path=C:\Users\sglvladi\Documents\TensorFlow\workspace\training_demo\images\test
```
  
  #### Configure training
  
  Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the faster_rcnn_inception_v2_pets.config (ตามโมเดลที่เลือกโหลดจาก model zoo) file
  แก้ไขบางบรรทัดต่อไปนี้ (เลขบรรทัดอาจะคลาดเคลื่อน)
  ```
  Line 9. Change num_classes to the number of different objects you want the classifier to detect.
  ```
```
Line 106. Change fine_tune_checkpoint to:

fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
```
```
Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow1/models/research/object_detection/train.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
```
```
Line 130. Change num_examples to the number of images you have in the \images\test directory.
```
```
Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow1/models/research/object_detection/test.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
```
  
### run train

```
python model_main.py --alsologtostderr --model_dir=training/ --pipeline_config_path=training/[model_name].config
```

### Exporting a Trained Inference Graph

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-13302 --output_directory trained-inference-graphs/output_inference_graph_v1.pb
```

  
