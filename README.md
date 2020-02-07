# Train-an-Object-Detection-Classifier 

## Youtube tutorial
[![Train-an-Object-Detection-Classifier](https://img.youtube.com/vi/eJcIntjgCbQ/0.jpg)](https://youtu.be/eJcIntjgCbQ)

### 1 - Gathering images
in my case I use google images to find [chinook] photos , I gather 200 sample

### 2 - Convert extensoins 
convert all images extensoins to `xx.jpg` , using Format Factory http://www.pcfreetime.com/

### 3 - Rename all images 
go to images directory in my case `chinook` and type in cmd `python renameFiles.py`

### 4 - Label images 
using `labelImg` https://github.com/tzutalin/labelImg
unzip labelImg\
run cmd and go to labelImg dir
```
conda install pyqt=5 
pyrcc5 -o resources.py resources.qrc
python labelImg.py

  ```
   
### 5 - Split images manually randomly
all my samples = 200 ,130 for train ,70 for test

### 6 - Installing TensorFlow-GPU

```pip install tensorflow-gpu 
pip install --upgrade tensorflow-gpu
```

### 7 - Creat virtual environment 

```
conda create -n tensorflow1 pip python=3.5 
activate tensorflow1 
pip install --ignore-installed --upgrade tensorflow-gpu

```
  other necessary packages
```
(tensorflow1) C:\> conda install -c anaconda protobuf 
(tensorflow1) C:\> pip install pillow 
(tensorflow1) C:\> pip install lxml 
(tensorflow1) C:\> pip install Cython 
(tensorflow1) C:\> pip install jupyter 
(tensorflow1) C:\> pip install matplotlib 
(tensorflow1) C:\> pip install pandas 
(tensorflow1) C:\> pip install opencv-python 
```
### 8 - Download the full TensorFlow object detection repository
https://github.com/tensorflow/models.git


### 9 - Download  `faster_rcnn_inception_v2_coco`
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

### 10 - Download  my Repository 
https://github.com/seraj94ai/Train-Object-Detection-Classifier.git
unzip folder and copy paste in `C:\tensorflow1\models\research\object_detection`
open cmd
```
cd C:\tensorflow1\models\research\object_detection
mkdir images
mkdir inference_graph
mkdir training
```

### 11 - Configure environment variable

Configure PYTHONPATH environment variable

PYTHONPATH variable must be created that points to the directories
\models \
\models\research \
\models\research\slim  
##### NOTE : every time you run your project must add this lines
```
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
echo %PYTHONPATH%
set PATH=%PATH%;PYTHONPATH
echo %PATH%
```
### 12 - Compile Protobufs
Protobuf (Protocol Buffers) libraries must be compiled , it used by TensorFlow to configure model and training parameters
Open Anaconda Prompt and go to `C:\tensorflow1\models\research`
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
```
```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```
### 13 - Test TensorFlow setup
Test TensorFlow setup to verify it works
`(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb`

### 14 - Generate Training Data
TFRecords is an input data to the TensorFlow training model
creat `.csv` files from `.xml` files 
```
cd C:\tensorflow1\models\research\object_detection
python xml_to_csv.py
```
This creates a `train_labels.csv` and `test_labels.csv` file in the `\object_detection\images` folder.
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
### 15 - Edit `generate_tfrecord.py`
edit edit `generate_tfrecord.py` and put your classes names

### 16 - Create a label map and edit the training configuration file.
go to `\data `
copy `pet_label_map.pbtxt` to `\training` dir and rename it to  `labelmap.pbtxt`

edit it  to your class `chinook`

### 17 - Configure object detection tranning pipeline
`cd C:\tensorflow1\models\research\object_detection\samples\configs`
copy `faster_rcnn_inception_v2_pets.config`
past it in  `\training` dir and edit it 


#### a - 
 In the `model` section change `num_classes` to number of different classes

#### b - 
 fine_tune_checkpoint : `C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt`

#### c - 
In the `train_input_reader` section change `input_path` and `label_map_path` as : <br/>
Input_path : `C:/tensorflow1/models/research/object_detection/train.record` <br/>
Label_map_path: `C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt`

#### d - 
In the `eval_config` section change `num_examples` as : <br/>
Num_examples = number of  files in   `\images\test` directory.

#### e -
In the `eval_input_reader` section change `input_path` and `label_map_path` as :<br/>
Input_path : `C:/tensorflow1/models/research/object_detection/test.record` <br/>
Label_map_path: `C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt`

### 18 - Run the Training
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

### 19 - Tensorboard 
in cmd type `(tensorflow1) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training`


### 20 - Export Inference Graph
 training is complete ,the last step is to generate the frozen inference graph (.pb file)
change “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```


![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/in/2.jpg)
![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/out/2.PNG)
 
![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/in/3.jpg)
![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/out/3.PNG)

![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/in/4.jpg)
![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/out/4.PNG)

![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/in/5.jpg)
![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/out/5.PNG)

![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/in/1.jpg)
![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/out/1.PNG)



# Appendix: Common Errors
#### 1. ModuleNotFoundError: No module named 'deployment'

![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/errs/err3.PNG)

![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/errs/solve%20err3.PNG)

You can use “echo %PATH%” and “echo %PYTHONPATH%” to check the environment variables and make sure they are set up correctly.
Also, make sure you have run these commands from the `\models\research` directory:
```
setup.py build
setup.py install
```

#### 2. ImportError: cannot import name 'preprocessor_pb2'
ImportError: cannot import name 'string_int_label_map_pb2'
(or similar errors with other pb2 files)
This occurs when the protobuf files (in this case, preprocessor.proto) have not been compiled
go to step 12

#### 3.
![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/errs/err1.PNG)

![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/errs/solve%20err1.PNG)
in `C:\tensorflow1\models\research\object_detection\utils\learning_schedules.py`

replace `range(num_boundaries)` to  `[i for i in range(num_boundaries)]`
```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```

#### 4.
![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/errs/err2.PNG)

![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/errs/solve%20err2.PNG)

![alt text](https://github.com/seraj94ai/Train-Object-Detection-Classifier/blob/master/errs/solve%20err%202.PNG)

in `C:\tensorflow1\models\research\object_detection\generate_tfrecord.py`
replace `[else: None]` by `[else: return 0]`














