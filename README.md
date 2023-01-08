# Project: Object Detection in an Urban Environment
<br/><br/>
## 1. Project overview
- Description of the Project
Detection is an algorithm that combines Classification and Localization of objects in an image.<br/>
The purpose of this project is to improve the detection performance of the ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 model in Urban Environment using the TFRecord file provided by Waymo Open Dataset.<br/>
- What we are trying to achieve
Today, Perception plays a very important role in the autonomous driving industry.<br/>
Therefore, we need to design a detection model with high accuracy and fast Inference performance.<br/>
In the real Urban Environment, there are a lot of vehicles and pedestrians.<br/>
In this situation, we proceeded with this project to design a safe and efficient perception model using Deep Learning.<br/><br/>

## 2. Set up
1. Training
```{.bash}
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
2. Evaluating
```{.bash}
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```
4. Monioring process
```{.bash}
python -m tensorboard.main --logdir experiments/reference/
```
5. Export trained model
```{.bash}
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```
6. Inference
```{.bash}
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## 3. Dataset
[Waymo Open Dataset](https://waymo.com/open/) <br/>
<img width="200" alt="image" src="https://user-images.githubusercontent.com/98406354/211144574-aec7f340-f9fd-4bb4-a59d-62c4fa25fe3e.png">
<br/>
The waymo open data set has image data taken from various urban environments.<br/>
The images below represent the representative characteristics of the data set.<br/>

<img width="300" alt="image" src="https://user-images.githubusercontent.com/98406354/211184077-c9ad2e19-ee7c-4e26-b047-e1588b49a235.png">
It is an image of a general city road situation.<br/>
There are many vehicles waiting for a driving signal in front of the waymo vehicle, and pedestrians standing on the sidewalk.
<img width="300" alt="image" src="https://user-images.githubusercontent.com/98406354/211184074-09f29d6c-b1ab-4e42-96ff-e18478e00b60.png">
The lens of the Camera sensor is contaminated due to external factors such as rain and snow, and a distorted image is captured.
<img width="300" alt="image" src="https://user-images.githubusercontent.com/98406354/211184078-6c974905-cb53-426c-bd0d-369e1ab3e56c.png">
It is a case with a low RGB Pixel value at a late night.
<br/>

### The Needs of Data Augmentation

What can be seen from the above cases is that the image that the actual vehicle can acquire may not be provided with clear image quality such as first image.<br/>
In fact, in the case of TFRecord provided by this project, it was confirmed that about 15 out of 100 data sets include Occlusion and Darkness.<br/>
Therefore, in order to design a robust model for the above image distortion before actual training, data augmentation should be carried out to increase the number of images corresponding to the above case.<br/>

### Cross-Validation
It is necessary to check whether the weight of the model obtained through training can produce effective performance even if a completely different image is input during the actual reference.<br/>
We can do this using Cross-validation, and in the case of this project, the provided TFRecord data set was used by data_split as 'Train: 80%, Validation: 10%, Test: 10%.<br/><br/>

## 4. Training

### Reference experiment
############################ 레퍼런스 실험의 Total Loss를 안 찍었네 3.88 몇 나왔는데 그거 다시 찍으려면  Pipeline-new.config에서 Data augmentation 다시 initialize하고 돌려봐야함
![Untitled (2)](https://user-images.githubusercontent.com/98406354/211185612-166ae41c-5b7e-4d43-a2dd-87805996b05d.png)

![animation1](https://user-images.githubusercontent.com/98406354/211185615-117d3260-3b20-4412-83af-cceff6e7534b.gif)

### Improve on the Reference
As described above, since only about 15% of the images in Train dataset contain Occlusion and Darkness, it was determined that the corresponding cases could not be learned in the model only with Original Dataset, so I added the 'random_adjusted_hue' and then proceeded learning process.
