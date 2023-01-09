# Project: Object Detection in an Urban Environment

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

![label1](https://user-images.githubusercontent.com/98406354/211246506-d8c508f8-aef2-4a72-9dc6-d509243ba6cf.png)
![label23](https://user-images.githubusercontent.com/98406354/211246515-18cac0b1-1110-43d9-9a25-69dfdbc759a3.png)

Through some modifications in 'Explanatory Data Analysis.ipynb', the Ground Truth bounding box of Objects in the image was plotted.

### The Needs of Data Augmentation

What can be seen from the above cases is that the image that the actual vehicle can acquire may not be provided with clear image quality such as first image.<br/>
In fact, in the case of TFRecord provided by this project, it was confirmed that about 15 out of 100 data sets include Occlusion and Darkness.<br/>
Therefore, in order to design a robust model for the above image distortion before actual training, data augmentation should be carried out to increase the number of images corresponding to the above case.<br/>

### Cross-Validation
It is necessary to check whether the weight of the model obtained through training can produce effective performance even if a completely different image is input during the actual reference.<br/>
We can do this using Cross-validation, and in the case of this project, the provided TFRecord data set was used by data_split as 'Train: 80%, Validation: 10%, Test: 10%.<br/><br/>

## 4. Training

### Reference experiment

![캡처](https://user-images.githubusercontent.com/98406354/211243711-5d993acd-c0ff-4a39-bc1b-4f328785e42c.PNG)

![asdf](https://user-images.githubusercontent.com/98406354/211243669-d86f9040-6768-48b9-a2c2-b5640aefef2e.gif)

I don't know the reason, but as a result of the reference experience, it was confirmed that the object in the image was not well detected as shown in the gif file below.



### Improve on the Reference

![Untitled (2)](https://user-images.githubusercontent.com/98406354/211185612-166ae41c-5b7e-4d43-a2dd-87805996b05d.png)

![animation1](https://user-images.githubusercontent.com/98406354/211185615-117d3260-3b20-4412-83af-cceff6e7534b.gif)

As described above, since only about 15% of the images in Train dataset contain Occlusion and Darkness, it was determined that the corresponding cases could not be learned in the model only with Original Dataset, so I added the 'random_adjusted_hue' and then proceeded learning process.

## Discussion
When first learning was attempted, many changes were made, such as adding a large amount of data augmentation methods and increasing learning steps. However, such a method has made the performance of models worse.<br/>
Through this, I found that applying an inappropriate Data Augmentation methods or making many changes to the model at once could have a worse effect on learning.<br/><br/>
First of all, it was necessary to understand the given training data set before modifying the Pipeline of the model.<br/>
The Training Dataset used in this project included images that could be classified as Distorted images, such as Accusion and Darkness, as well as captured images with clear image quality as described above.<br/>
Therefore, during model learning, I judged that I should learn the image of cases as above to produce better Inference performance.<br/><br/>
Early in the experiment, various data augmentation methods were applied, which was rather unnecessary modification.<br/>
I found out that there were few dark images (including Darkness + Occlusion) among the training datasets. So I applied 'random_adjusted_hue' to learn.<br/>
As a result, as can be seen in the 'Improve on the Reference' section, the model was able to perform better during the Inference.<br/><br/>
What I learned from this is that it is very important to have a high understanding of Dataset.<br/>
Previously, I thought that doing as much Data Augmentation as possible was a way to obtain model weights suitable for various Datasets, but experiments have shown that such methods can sometimes degrade the performance of models.<br/>
Therefore, before conducting the experiment, I found that it is a very important process to grasp the feature of Dataset and select the appropriate Data Augmentation methods.<br/>
