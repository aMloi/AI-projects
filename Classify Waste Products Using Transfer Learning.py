# [**Final Project: Classify Waste Products Using Transfer Learning**](#toc0_)




**Table of contents**<a id='toc0_'></a>    
- [**Classify Waste Products Using Transfer Learning**](#toc1_)    
  - [Introduction](#toc1_1_)    
    - [Project Overview](#toc1_1_1_)    
    - [Aim of the Project](#toc1_1_2_)    
  
  - [Setup](#toc1_3_)    
    - [Installing Required Libraries](#toc1_3_1_)    
    - [Importing Required Libraries](#toc1_3_2_)    


### <a id='toc1_1_1_'></a>[Project Overview](#toc0_)

EcoClean currently lacks an efficient and scalable method to automate the waste sorting process. The manual sorting of waste is not only labor-intensive but also prone to errors, leading to contamination of recyclable materials. The goal of this project is to leverage machine learning and computer vision to automate the classification of waste products, improving efficiency and reducing contamination rates. The project will use transfer learning with a pre-trained VGG16 model to classify images.

### <a id='toc1_1_2_'></a>[Aim of the Project](#toc0_)

The aim of the project is to develop an automated waste classification model that can accurately differentiate between recyclable and organic waste based on images.

**Final Output**: A trained model that classifies waste images into recyclable and organic categories.




```python
!pip install tensorflow==2.17.0 | tail -n 1
!pip install numpy==1.24.3 | tail -n 1
!pip install scikit-learn==1.5.1  | tail -n 1
!pip install matplotlib==3.9.2  | tail -n 1
```

    Successfully installed absl-py-2.2.2 astunparse-1.6.3 flatbuffers-25.2.10 gast-0.6.0 google-pasta-0.2.0 grpcio-1.71.0 h5py-3.13.0 keras-3.9.2 libclang-18.1.1 markdown-3.7 markdown-it-py-3.0.0 mdurl-0.1.2 ml-dtypes-0.4.1 namex-0.0.8 numpy-1.26.4 opt-einsum-3.4.0 optree-0.15.0 protobuf-4.25.6 rich-14.0.0 tensorboard-2.17.1 tensorboard-data-server-0.7.2 tensorflow-2.17.0 termcolor-3.0.1 werkzeug-3.1.3 wrapt-1.17.2
      [1;31merror[0m: [1msubprocess-exited-with-error[0m
      
      [31m×[0m [32mGetting requirements to build wheel[0m did not run successfully.
      [31m│[0m exit code: [1;36m1[0m
      [31m╰─>[0m [31m[33 lines of output][0m
      [31m   [0m Traceback (most recent call last):
      [31m   [0m   File "/opt/conda/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
      [31m   [0m     main()
      [31m   [0m   File "/opt/conda/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
      [31m   [0m     json_out["return_val"] = hook(**hook_input["kwargs"])
      [31m   [0m                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      [31m   [0m   File "/opt/conda/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 137, in get_requires_for_build_wheel
      [31m   [0m     backend = _build_backend()
      [31m   [0m               ^^^^^^^^^^^^^^^^
      [31m   [0m   File "/opt/conda/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 70, in _build_backend
      [31m   [0m     obj = import_module(mod_path)
      [31m   [0m           ^^^^^^^^^^^^^^^^^^^^^^^
      [31m   [0m   File "/opt/conda/lib/python3.12/importlib/__init__.py", line 90, in import_module
      [31m   [0m     return _bootstrap._gcd_import(name[level:], package, level)
      [31m   [0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      [31m   [0m   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
      [31m   [0m   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
      [31m   [0m   File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
      [31m   [0m   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
      [31m   [0m   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
      [31m   [0m   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
      [31m   [0m   File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
      [31m   [0m   File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
      [31m   [0m   File "<frozen importlib._bootstrap_external>", line 999, in exec_module
      [31m   [0m   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
      [31m   [0m   File "/tmp/pip-build-env-v40bol7g/overlay/lib/python3.12/site-packages/setuptools/__init__.py", line 16, in <module>
      [31m   [0m     import setuptools.version
      [31m   [0m   File "/tmp/pip-build-env-v40bol7g/overlay/lib/python3.12/site-packages/setuptools/version.py", line 1, in <module>
      [31m   [0m     import pkg_resources
      [31m   [0m   File "/tmp/pip-build-env-v40bol7g/overlay/lib/python3.12/site-packages/pkg_resources/__init__.py", line 2172, in <module>
      [31m   [0m     register_finder(pkgutil.ImpImporter, find_on_path)
      [31m   [0m                     ^^^^^^^^^^^^^^^^^^^
      [31m   [0m AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?
      [31m   [0m [31m[end of output][0m
      
      [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
    [1;31merror[0m: [1msubprocess-exited-with-error[0m
    
    [31m×[0m [32mGetting requirements to build wheel[0m did not run successfully.
    [31m│[0m exit code: [1;36m1[0m
    [31m╰─>[0m See above for output.
    
    [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
      Getting requirements to build wheel: finished with status 'error'
    Successfully installed joblib-1.4.2 scikit-learn-1.5.1 scipy-1.15.2 threadpoolctl-3.6.0
    Successfully installed contourpy-1.3.1 cycler-0.12.1 fonttools-4.57.0 kiwisolver-1.4.8 matplotlib-3.9.2 pillow-11.1.0 pyparsing-3.2.3


### <a id='toc1_3_2_'></a>[Importing Required Libraries](#toc0_)



```python
import numpy as np
import os
import glob


from matplotlib import pyplot as plt

from matplotlib.image import imread

from os import makedirs,listdir
from shutil import copyfile
from random import seed
from random import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

```


```python
tf.__version__
```




    '2.17.0'




```python
import requests
import zipfile
from tqdm import tqdm

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
file_name = "o-vs-r-split-reduced-1200.zip"

print("Downloading file")
with requests.get(url, stream=True) as response:
    response.raise_for_status()
    with open(file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def extract_file_with_progress(file_name):
    print("Extracting file with progress")
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        members = zip_ref.infolist() 
        with tqdm(total=len(members), unit='file') as progress_bar:
            for member in members:
                zip_ref.extract(member)
                progress_bar.update(1)
    print("Finished extracting file")


extract_file_with_progress(file_name)

print("Finished extracting file")
os.remove(file_name)
```

    Downloading file
    Extracting file with progress


    100%|██████████| 1207/1207 [01:53<00:00, 10.67file/s]

    Finished extracting file
    Finished extracting file


    


### <a id='toc1_6_3_'></a>[Define configuration options](#toc0_)
 define some model configuration options.

*   **batch size** is set to 32.
*   The **number of classes** is 2.
*   use 20% of the data for **validation** purposes.
*   two **labels** in your dataset: organic (O), recyclable (R).



```python
img_rows, img_cols = 150, 150
batch_size = 32
n_epochs = 10
n_classes = 2
val_split = 0.2
verbosity = 1
path = 'o-vs-r-split/train/'
path_test = 'o-vs-r-split/test/'
input_shape = (img_rows, img_cols, 3)
labels = ['O', 'R']
seed = 42
```


```python
# Creating ImageDataGenerators for training and validation and testing
train_datagen = ImageDataGenerator(
    validation_split = val_split,
    rescale=1.0/255.0,
	width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    validation_split = val_split,
    rescale=1.0/255.0,
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255.0
)
```


```python
train_generator = train_datagen.flow_from_directory(
    directory = path,
    seed = seed,
    batch_size = batch_size, 
    class_mode='binary',
    shuffle = True,
    target_size=(img_rows, img_cols),
    subset = 'training'
)
```

    Found 800 images belonging to 2 classes.



```python
val_generator = val_datagen.flow_from_directory(
    directory = path,
    seed = seed,
    batch_size = batch_size, 
    class_mode='binary',
    shuffle = True,
    target_size=(img_rows, img_cols),
    subset = 'validation'
)
```

    Found 200 images belonging to 2 classes.



```python
# Task 2: Create a `test_generator` using the `test_datagen` object
test_generator = test_datagen.flow_from_directory(   directory= path ,
    class_mode= 'binary' ,
    seed= seed,
    batch_size= batch_size,
    shuffle= True,
    target_size= (img_rows, img_cols)
                                                 )
```

    Found 1000 images belonging to 2 classes.



```python
len(train_generator)
```




    25




```python
from pathlib import Path

IMG_DIM = (100, 100)

train_files = glob.glob('./o-vs-r-split/train/O/*')
train_files = train_files[:20]
train_imgs = [tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [Path(fn).parent.name for fn in train_files]

img_id = 0
O_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1],
                                   batch_size=1)
O = [next(O_generator) for i in range(0,5)]
fig, ax = plt.subplots(1,5, figsize=(16, 6))
print('Labels:', [item[1][0] for item in O])
l = [ax[i].imshow(O[i][0][0]) for i in range(0,5)]

```

    Labels: ['O', 'O', 'O', 'O', 'O']



    
![png](output_14_1.png)
    


### <a id='toc1_8_1_'></a>[Pre-trained Models](#toc0_)

Pre-trained models are saved networks that have previously been trained on some large datasets. They are typically used for large-scale image-classification task. They can be used as they are or could be customized to a given task using transfer learning. These pre-trained models form the basis of transfer learning.

#### <a id='toc1_8_1_1_'></a>[VGG-16](#toc0_)

Lets load the VGG16 model.



```python
from tensorflow.keras.applications import vgg16

input_shape = (150, 150, 3)
vgg = vgg16.VGG16(include_top=False,
                        weights='imagenet',
                        input_shape=input_shape)


```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    [1m58889256/58889256[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step



```python
output = vgg.layers[-1].output
output = tf.keras.layers.Flatten()(output)
basemodel = Model(vgg.input, output)
```

Next, we freeze the basemodel.



```python
for layer in basemodel.layers: 
    layer.trainable = False
```


```python
input_shape = basemodel.output_shape[1]

model = Sequential()
model.add(basemodel)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
```


```python
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ functional (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8192</span>)           │    <span style="color: #00af00; text-decoration-color: #00af00">14,714,688</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │     <span style="color: #00af00; text-decoration-color: #00af00">4,194,816</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">262,656</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">513</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">19,172,673</span> (73.14 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,457,985</span> (17.01 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">14,714,688</span> (56.13 MB)
</pre>




```python
for layer in basemodel.layers: 
    layer.trainable = False

# Task 5: Compile the model
model.compile(
    loss= 'binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics= ['accuracy']
)
```

 will use early stopping to avoid over-training the model.



```python
from tensorflow.keras.callbacks import LearningRateScheduler


checkpoint_path='O_R_tlearn_vgg16.keras'

# define step decay function
class LossHistory_(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(epoch))
        print('lr:', exp_decay(len(self.losses)))

def exp_decay(epoch):
    initial_lrate = 1e-4
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate

# learning schedule callback
loss_history_ = LossHistory_()
lrate_ = LearningRateScheduler(exp_decay)

keras_callbacks = [
      EarlyStopping(monitor = 'val_loss', 
                    patience = 4, 
                    mode = 'min', 
                    min_delta=0.01),
      ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
]

callbacks_list_ = [loss_history_, lrate_] + keras_callbacks
```

## <a id='toc1_11_'></a>[Fit and train the model](#toc0_)



```python
extract_feat_model = model.fit(train_generator, 
                               steps_per_epoch=5, 
                               epochs=10,
                               callbacks = callbacks_list_,   
                               validation_data=val_generator, 
                               validation_steps=val_generator.samples // batch_size, 
                               verbose=1)
```

    Epoch 1/10
    lr: 9.048374180359596e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.4420 - loss: 0.8274 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 13s/step - accuracy: 0.4506 - loss: 0.8167 - val_accuracy: 0.8229 - val_loss: 0.5542 - learning_rate: 1.0000e-04
    Epoch 2/10
    lr: 8.187307530779819e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.6674 - loss: 0.5879 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 12s/step - accuracy: 0.6718 - loss: 0.5853 - val_accuracy: 0.7135 - val_loss: 0.5220 - learning_rate: 9.0484e-05
    Epoch 3/10
    lr: 7.408182206817179e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.7327 - loss: 0.4919 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 19s/step - accuracy: 0.7325 - loss: 0.4941 - val_accuracy: 0.7969 - val_loss: 0.4533 - learning_rate: 8.1873e-05
    Epoch 4/10
    lr: 6.703200460356393e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.7616 - loss: 0.4611 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 12s/step - accuracy: 0.7576 - loss: 0.4678 - val_accuracy: 0.8542 - val_loss: 0.4106 - learning_rate: 7.4082e-05
    Epoch 5/10
    lr: 6.065306597126335e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8053 - loss: 0.4710 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 11s/step - accuracy: 0.8044 - loss: 0.4686 - val_accuracy: 0.7865 - val_loss: 0.4294 - learning_rate: 6.7032e-05
    Epoch 6/10
    lr: 5.488116360940264e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4s/step - accuracy: 0.7965 - loss: 0.4305 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 12s/step - accuracy: 0.7970 - loss: 0.4313 - val_accuracy: 0.8490 - val_loss: 0.3745 - learning_rate: 6.0653e-05
    Epoch 7/10
    lr: 4.9658530379140954e-05━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.7108 - loss: 0.5159 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 12s/step - accuracy: 0.7194 - loss: 0.5066 - val_accuracy: 0.8646 - val_loss: 0.3648 - learning_rate: 5.4881e-05
    Epoch 8/10
    lr: 4.493289641172216e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8249 - loss: 0.3662 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m49s[0m 11s/step - accuracy: 0.8197 - loss: 0.3720 - val_accuracy: 0.8177 - val_loss: 0.3857 - learning_rate: 4.9659e-05
    Epoch 9/10
    lr: 4.0656965974059915e-05━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4s/step - accuracy: 0.8336 - loss: 0.3984 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 11s/step - accuracy: 0.8343 - loss: 0.3974 - val_accuracy: 0.8542 - val_loss: 0.3443 - learning_rate: 4.4933e-05
    Epoch 10/10
    lr: 3.678794411714424e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8373 - loss: 0.3647 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 12s/step - accuracy: 0.8321 - loss: 0.3688 - val_accuracy: 0.8646 - val_loss: 0.3354 - learning_rate: 4.0657e-05


### <a id='toc1_11_1_'></a>[Plot loss curves for training and validation sets (extract_feat_model)](#toc0_)



```python
import matplotlib.pyplot as plt

history = extract_feat_model

# plot loss curve
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```


    
![png](output_28_0.png)
    



```python
history = extract_feat_model
plt.figure(figsize=(5, 5))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()
```


    
![png](output_29_0.png)
    


## <a id='toc1_12_'></a>[Fine-Tuning model](#toc0_)





```python
from tensorflow.keras.applications import vgg16

input_shape = (150, 150, 3)
vgg = vgg16.VGG16(include_top=False,
                        weights='imagenet',
                        input_shape=input_shape)

output = vgg.layers[-1].output
output = tf.keras.layers.Flatten()(output)
basemodel = Model(vgg.input, output)

for layer in basemodel.layers: 
    layer.trainable = False

display([layer.name for layer in basemodel.layers])

set_trainable = False

for layer in basemodel.layers:
    if layer.name in ['block5_conv3']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

for layer in basemodel.layers:
    print(f"{layer.name}: {layer.trainable}")
```


    ['input_layer_2',
     'block1_conv1',
     'block1_conv2',
     'block1_pool',
     'block2_conv1',
     'block2_conv2',
     'block2_pool',
     'block3_conv1',
     'block3_conv2',
     'block3_conv3',
     'block3_pool',
     'block4_conv1',
     'block4_conv2',
     'block4_conv3',
     'block4_pool',
     'block5_conv1',
     'block5_conv2',
     'block5_conv3',
     'block5_pool',
     'flatten_1']


    input_layer_2: False
    block1_conv1: False
    block1_conv2: False
    block1_pool: False
    block2_conv1: False
    block2_conv2: False
    block2_pool: False
    block3_conv1: False
    block3_conv2: False
    block3_conv3: False
    block3_pool: False
    block4_conv1: False
    block4_conv2: False
    block4_conv3: False
    block4_pool: False
    block5_conv1: False
    block5_conv2: False
    block5_conv3: True
    block5_pool: True
    flatten_1: True



```python
model = Sequential()
model.add(basemodel)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

checkpoint_path='O_R_tlearn_fine_tune_vgg16.keras'

# learning schedule callback
loss_history_ = LossHistory_()
lrate_ = LearningRateScheduler(exp_decay)

keras_callbacks = [
      EarlyStopping(monitor = 'val_loss', 
                    patience = 4, 
                    mode = 'min', 
                    min_delta=0.01),
      ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
]

callbacks_list_ = [loss_history_, lrate_] + keras_callbacks

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['accuracy'])

fine_tune_model = model.fit(train_generator, 
                    steps_per_epoch=5, 
                    epochs=10,
                    callbacks = callbacks_list_,   
                    validation_data=val_generator, 
                    validation_steps=val_generator.samples // batch_size, 
                    verbose=1)
```

    Epoch 1/10
    lr: 9.048374180359596e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.5655 - loss: 0.7175 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 12s/step - accuracy: 0.5723 - loss: 0.7117 - val_accuracy: 0.6771 - val_loss: 0.5741 - learning_rate: 1.0000e-04
    Epoch 2/10
    lr: 8.187307530779819e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.6370 - loss: 0.6266 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 12s/step - accuracy: 0.6506 - loss: 0.6165 - val_accuracy: 0.7969 - val_loss: 0.4632 - learning_rate: 9.0484e-05
    Epoch 3/10
    lr: 7.408182206817179e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.7135 - loss: 0.5528 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 12s/step - accuracy: 0.7196 - loss: 0.5487 - val_accuracy: 0.7865 - val_loss: 0.4339 - learning_rate: 8.1873e-05
    Epoch 4/10
    lr: 6.703200460356393e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8077 - loss: 0.4440 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 12s/step - accuracy: 0.8054 - loss: 0.4463 - val_accuracy: 0.8073 - val_loss: 0.3862 - learning_rate: 7.4082e-05
    Epoch 5/10
    lr: 6.065306597126335e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8036 - loss: 0.3911 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 12s/step - accuracy: 0.8051 - loss: 0.3894 - val_accuracy: 0.8073 - val_loss: 0.3602 - learning_rate: 6.7032e-05
    Epoch 6/10
    lr: 5.488116360940264e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8798 - loss: 0.2971 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 12s/step - accuracy: 0.8769 - loss: 0.3024 - val_accuracy: 0.8177 - val_loss: 0.3362 - learning_rate: 6.0653e-05
    Epoch 7/10
    lr: 4.9658530379140954e-05━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8869 - loss: 0.3134 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 12s/step - accuracy: 0.8818 - loss: 0.3181 - val_accuracy: 0.8646 - val_loss: 0.3089 - learning_rate: 5.4881e-05
    Epoch 8/10
    lr: 4.493289641172216e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8729 - loss: 0.3479 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 12s/step - accuracy: 0.8681 - loss: 0.3540 - val_accuracy: 0.8646 - val_loss: 0.3053 - learning_rate: 4.9659e-05
    Epoch 9/10
    lr: 4.0656965974059915e-05━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8441 - loss: 0.3511 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 12s/step - accuracy: 0.8430 - loss: 0.3535 - val_accuracy: 0.8646 - val_loss: 0.3161 - learning_rate: 4.4933e-05
    Epoch 10/10
    lr: 3.678794411714424e-05━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.8737 - loss: 0.3314 
    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 13s/step - accuracy: 0.8677 - loss: 0.3364 - val_accuracy: 0.8698 - val_loss: 0.2895 - learning_rate: 4.0657e-05



```python
history = fine_tune_model

plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```


    
![png](output_33_0.png)
    



```python
history = fine_tune_model

plt.figure(figsize=(5, 5))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[37], line 4
          1 history = fine_tune_model
          3 plt.figure(figsize=(5, 5))
    ----> 4 plt.plot(history.history['accuracy'], label='Training accuracy')
          5 plt.plot(history.history['val_accuracy'], label='Validation accuracy')
          6 plt.title('accuracy Curve')


    AttributeError: 'Sequential' object has no attribute 'history'



    <Figure size 500x500 with 0 Axes>


## <a id='toc1_13_'></a>[Evaluate both models on test data](#toc0_)

- Load saved models
- Load test images
- Make predictions for both models
- Convert predictions to class labels
- Print classification report for both models



```python
from pathlib import Path

# Load saved models
extract_feat_model = tf.keras.models.load_model('O_R_tlearn_vgg16.keras')
fine_tune_model = tf.keras.models.load_model('O_R_tlearn_fine_tune_vgg16.keras')

IMG_DIM = (150, 150)

# Load test images
test_files_O = glob.glob('./o-vs-r-split/test/O/*')
test_files_R = glob.glob('./o-vs-r-split/test/R/*')
test_files = test_files_O[:50] + test_files_R[:50]

test_imgs = [tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
test_labels = [Path(fn).parent.name for fn in test_files]

# Standardize
test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255

class2num_lt = lambda l: [0 if x == 'O' else 1 for x in l]
num2class_lt = lambda l: ['O' if x < 0.5 else 'R' for x in l]

test_labels_enc = class2num_lt(test_labels)

# Make predictions for both models
predictions_extract_feat_model = extract_feat_model.predict(test_imgs_scaled, verbose=0)
predictions_fine_tune_model = fine_tune_model.predict(test_imgs_scaled, verbose=0)

# Convert predictions to class labels
predictions_extract_feat_model = num2class_lt(predictions_extract_feat_model)
predictions_fine_tune_model = num2class_lt(predictions_fine_tune_model)

# Print classification report for both models
print('Extract Features Model')
print(metrics.classification_report(test_labels, predictions_extract_feat_model))
print('Fine-Tuned Model')
print(metrics.classification_report(test_labels, predictions_fine_tune_model))

```

    Extract Features Model
                  precision    recall  f1-score   support
    
               O       0.75      0.88      0.81        50
               R       0.85      0.70      0.77        50
    
        accuracy                           0.79       100
       macro avg       0.80      0.79      0.79       100
    weighted avg       0.80      0.79      0.79       100
    
    Fine-Tuned Model
                  precision    recall  f1-score   support
    
               O       0.75      0.86      0.80        50
               R       0.84      0.72      0.77        50
    
        accuracy                           0.79       100
       macro avg       0.80      0.79      0.79       100
    weighted avg       0.80      0.79      0.79       100
    



```python
def plot_image_with_title(image, model_name, actual_label, predicted_label):
    plt.imshow(image)
    plt.title(f"Model: {model_name}, Actual: {actual_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

index_to_plot = 0
plot_image_with_title(
    image=test_imgs[index_to_plot].astype('uint8'),
    model_name='Extract Features Model',
    actual_label=test_labels[index_to_plot], 
    predicted_label=predictions_extract_feat_model[index_to_plot],
    )
```


    
![png](output_37_0.png)
    


### <a id='toc1_13_1_'></a>[**Plot a test image using Extract Features Model**](#toc0_)


```python
index_to_plot = 1
plot_image_with_title(
    image=test_imgs[index_to_plot].astype('uint8'),
    model_name='Extract Features Model',
    actual_label=test_labels[index_to_plot], 
    predicted_label=predictions_extract_feat_model[index_to_plot],
    )
```


    
![png](output_39_0.png)
    


### <a id='toc1_13_2_'></a>[**Plot a test image using Fine-Tuned Model**](#toc0_)




```python
index_to_plot = 1
plot_image_with_title(
    image=test_imgs[index_to_plot].astype('uint8'),
    model_name='Fine-Tuned Model',
    actual_label=test_labels[index_to_plot], 
    predicted_label=predictions_fine_tune_model[index_to_plot],
    )
```


    
![png](output_41_0.png)
    

