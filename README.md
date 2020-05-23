# Deploy-a-People-Counter-App-at-the-Edge

![](Screenshot.png)

For this project, you’ll first find a useful person detection model and convert it to an Intermediate Representation for use with the Model Optimizer. Utilizing the Inference Engine, you'll use the model to perform inference on an input video, and extract useful data concerning the count of people in frame and how long they stay in frame. You'll send this information over MQTT, as well as sending the output frame, in order to view it from a separate UI server over a network.
