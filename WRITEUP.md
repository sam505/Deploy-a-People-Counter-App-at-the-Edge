# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name] -ssd_mobilenet_v2_coco_2018_03_29
  - [Model Source] -http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

  - I converted the model to an Intermediate Representation with the following arguments...
1. cd [Download directory] navigate to the doanload directory
2. tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz # Extract model file downloaded
3. cd ssd_mobilenet_v2_coco_2018_03_29/ # Enter the folder the downloaded model was extrated to
4. python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json --input_shape [1,300,300,3]


  - The model was insufficient for the app because...
Accuracy of the model was low when detecting a person in a frame
  - I tried to improve the model for the app by...
Lowered the value of the probability threshold but ended up detecting multiple objects that were not persons in one frame.
  
- Model 2: [Name] -faster_rcnn_nas_coco_2018_01_28
  - [Model Source] -http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz

  - I converted the model to an Intermediate Representation with the following arguments...
1. cd [Download directory]
2. tar -xvf faster_rcnn_nas_coco_2018_01_28.tar.gz # Extract model file downloaded
3. cd faster_rcnn_nas_coco_2018_01_28/ Enter the folder that the model file was extracted to
4. python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --input_shape [1,1200,1200,3]
command to convert the pretrained model to Intermediate Representation

  - The model was insufficient for the app because...
1. Network has 2 inputs overall

  - I tried to improve the model for the app by...
1. Co..

- Model 3: [Name] - faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  - [Model Source] - http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

  - I converted the model to an Intermediate Representation with the following arguments...
1. cd [Download directory]
2. tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz # Extract model file downloaded
3. cd faster_rcnn_inception_v2_coco_2018_01_28/ # Enter the directory where the model has been extracted to
4.  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --input_shape [-1,600,600,3]
command to convert the pretrained model to Intermediate Representation

  - The model was insufficient for the app because...
1. Network has 2 inputs overall
2. On using the model in the app, I got an error that the model only receives two input topologies

  - I tried to improve the model for the app by...

- Model 4: [Name] - MobileNet-SSD
  - [Model Source] - https://codeload.github.com/chuanqi305/MobileNet-SSD

  - I converted the model lsto an Intermediate Representation with the following arguments...
1. cd [Download directory]
2. cd MobileNet-SSD-master/
3. python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model mobilenet_iter_73000.caffemodel --input_proto deploy.prototxt

  - The model was insufficient for the app because...
1. No accurate detections were made using the model
2. Detected not satisfied dependencies:
	protobuf: installed: 3.7.1, required: == 3.6.1

  - I tried to improve the model for the app by...

- Model 5: [Name] - 
  - [Model Source] - 

  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
