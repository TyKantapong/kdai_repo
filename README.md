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
