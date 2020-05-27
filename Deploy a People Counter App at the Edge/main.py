"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


# prints information about model layers
def performance_counts(perf_count):
    # perf_count is a dictionary containing the
    # status of the model status
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    # prob_threshold = args.prob_threshold
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    time_on_video = 0
    time_not_on_video = 0
    image_mode = False
    positive_count = 0
    ### TODO: Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1,
                                          cur_request_id, args.cpu_extension)[1]
    ### TODO: Handle the input stream ###
    # Checks for image input
    if args.input.endswith('.jpg') or args.input.endswith('.png') or \
            args.input.endswith('.bmp'):
        image_mode = True
        media_stream = args.input

    # Checks for webcam input
    elif args.input == 'CAM':
        media_stream = 0

    # Check for video input
    else:
        media_stream = args.input
        assert os.path.isfile(args.input)

    ### TODO: Loop until stream is over ###
    capture = cv2.VideoCapture(media_stream)

    if media_stream:
        capture.open(args.input)

    if not capture.isOpened():
        log.error("Not able to open the video file!")

        ### TODO: Read from the video capture ###
    # global width, height, prob_threshold
    prob_threshold = args.prob_threshold
    width = capture.get(3)
    height = capture.get(4)

    while capture.isOpened():
        check, frame = capture.read()
        if not check:
            break

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (w, h))
        image = image.transpose(2, 0, 1)
        image = image.reshape(n, c, h, w)

        ### TODO: Start asynchronous inference for specified request ###
        inference_start = time.time()
        infer_network.exec_net(cur_request_id, image)

        ### TODO: Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            inference_time = time.time() - inference_start

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(cur_request_id)

            # if perf_counts:
            # perf_count = infer_network.exec_net(cur_request_id)
            # performance_counts(perf_count)

            ### TODO: Extract any desired stats from the results ###
            current_count = 0
            track_frames = {}
            track_person = {positive_count: 0}
            frame_count = 0

            for character in result[0][0]:
                if character[2] > prob_threshold:
                    frame_count += 1
                    track_frames[frame_count] = character[2]
                    start_time_not_on_video = time.time()
                    positive_count += 1
                    track_person[positive_count] = time_on_video
                    xmin = int(character[3] * width)
                    ymin = int(character[4] * height)
                    xmax = int(character[5] * width)
                    ymax = int(character[6] * height)
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)

                    time_on_video = start_time_not_on_video - start_time
                    if time_on_video > 3:
                        if current_count > 1:
                            current_count = last_count
                        else:
                            current_count += 1
                    else:
                        current_count = last_count

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if current_count > last_count:
                start_time = time.time()
                time_not_on_video = time.time() - start_time_not_on_video
                if current_count == 1 and last_count == 0:
                    if time_on_video > 2:
                        total_count = total_count + current_count - last_count

            client.publish("person", json.dumps({"total": total_count}))
            if current_count < last_count:
                if current_count == 0:
                    start_time_not_on_video = time.time()
                time_on_video = int(time.time() - start_time)
                if last_count == 0 and time_not_on_video < 0.005:
                    time_on_video = track_person[positive_count] + time_on_video
                client.publish("person/duration", json.dumps({"duration": time_on_video}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

            cv2.putText(frame, "Inference time =  {:.2f} ms".format((inference_time * 1000)),
                        (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, "Persons in video frame = {:}".format(last_count), (15, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, "Total count = {:}".format(total_count), (15, 45),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, "Time on video = {:.2f} s".format(time_on_video), (15, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, "Time not on video = {:.3f} s".format(time_not_on_video * 1000), (15, 75),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            key = cv2.waitKey(15)
            if key == ord('q'):
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if image_mode:
            cv2.imwrite('output.jpg', frame)

        # cv2.imshow('frame', frame)

    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    cur_request_id = 0
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
