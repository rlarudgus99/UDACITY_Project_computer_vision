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

<img width="150" alt="image" src="https://user-images.githubusercontent.com/98406354/211144831-df131577-1936-4d7b-a9f5-66459971ed24.png">
<img width="150" alt="image" src="https://user-images.githubusercontent.com/98406354/211144831-df131577-1936-4d7b-a9f5-66459971ed24.png">
<img width="150" alt="image" src="https://user-images.githubusercontent.com/98406354/211144837-6ab6d29b-6b7c-451e-ac55-a549797d3087.png)">

