# Computer Pointer Controller

In this project I practice with difference models and precisions in order to create an application that allow move the cursor making an inference over the eyes of a person. Where the person is looking the mouse should be moved.

Using OpenVino (Intel) and other external python libraries I could go deeply in IA inference, taking in account the importance of the time, the hardware and the precision of the models.

## Project Set Up and Installation
### Downloading the models
You can download the models used in the application with the model downloader included  on the OpenVino Toolkit.  
Just use the following instructions on a terminal. (*You should be on the project folder.*)

`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 --output_dir ./models`

`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --output_dir ./models`

`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --output_dir ./models`

`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --output_dir ./models`

## Demo
You could find a small script just to make easy the first execution of the app. Just execute the following commands in a terminal:
```
source init.sh
./execute.sh
```

## Documentation
The application has some possible arguments that we are going to list:

```
usage: main.py [-h] -fdm FACE_DETECTION -flm LANDMARK_DETECTION_MODEL -hpm
               HEAD_POSE_MODEL -gem GAZE_ESTIMATION_MODEL -it INPUT_TYPE
               [-if INPUT_FILE] [-pt PROB_THRESHOLD] [-d DEBUG]

optional arguments:  
-h, --help            show this help message and exit  
  -fdm FACE_DETECTION, --face_detection FACE_DETECTION  
                        Path to face detection model.  
  -flm LANDMARK_DETECTION_MODEL, --landmark_detection_model LANDMARK_DETECTION_MODEL  
                        Path to landmark detection model.  
  -hpm HEAD_POSE_MODEL, --head_pose_model HEAD_POSE_MODEL  
                        Path to head pose estimation model.  
  -gem GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL  
                        Path to head gaze estimation model.  
  -it INPUT_TYPE, --input_type INPUT_TYPE  
                        Input type that is going to be used 'video', 'image' or 'cam'.  
  -if INPUT_FILE, --input_file INPUT_FILE  
                        Input file for using as inference file  
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD  
                        Probability threshold for detections faces(0.5 by default)  
  -d DEBUG, --debug DEBUG  
                        Debug mode print and write images  
```

## Benchmarks
### CPU
#### Load Time
| Precision       | Face Detection   | Landmarks Detection        | Headpose Estimation | Gaze Estimation |
|--------------------|---------------|-----------|-------------|-----------|
|FP32      |  198ms        | 51ms     | 62ms       | 84ms     |
|FP16      |  NA           | 55ms     | 80ms       | 97ms     |  
|FP16-INT8 |  NA           | 63ms     | 148ms      | 154ms     |

#### Inference Time
| Precision       | Face Detection   | Landmarks Detection        | Headpose Estimation | Gaze Estimation |
|--------------------|---------------|-----------|-------------|-----------|
|FP32       | 8.8ms         | 0.5ms     | 1.4ms       | 1.2ms     |
|FP16       | NA            | 0.6ms     | 1.6ms       | 1.2ms     |
|FP16-INT8  | NA            | 0.5ms     | 1.3ms       | 0.9ms     |

## Results
* Loading time is lower for more accurate models, so FP32 load faster than FP16-INT8
* Inference time is lightly faster in lower accurate models than in more precise models but in our case the time difference is not huge. Due to the use of a lower precision the inference need less time to be executed but the accuracy will be lower too.

## Stand Out Suggestions

### Async Inference
I'm using async inference in order to make more efficient the inference process. The face detection model has to perform it's inference sequentially but their output is used as input of both, landmark detection and head pose estimation models, so we could perform both inferences in parallel and for that reason I have to implement async inference. 

Executing in parallel this inferences we reduce the time needed to extract the information about the eyes positions and the head pose angles. The amount of time reduces will depend on the hardware that will be supporting the inference.

### Edge Cases
One of the edge cases that I found is threating with the eyes position and extraction. When the face is not complete centered in the cropped image, crop the eyes for the next inference could produce some problems with the coordinates so we have to deal with it. I have to crop the image taking in account the limits of the original image and after that resizing the cropped eye.

### Video and Webcam
The application works either with a video file input and the webcam. You could use the `INPUT_TYPE` parameter in order to select what kind of input do you want to use.

### Packaging the app

The ideal way to distribute is use the OpenVINO™ Deployment Manager but it's only supported by Linux or Windows. On the MacOS package is not present.

### Logging important events

I'm using try/except block to identify some errors or important events in the app execution and log it using `logging`. This makes easy the use of the application to a user when a problem occurred.