"# kdai_repo" 
# Train an Object Detection Classifier Using TensorFlow (GPU) on Windows 10

## Steps
### 1. Install Anaconda, CUDA, and cuDNN 
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

###Install TensorFlow GPU for Python
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


