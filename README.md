MOXGO - the ollama for ONNX
============================

get an easy http api for ONNX models on android,windows,linux,mac,ios with native acceleration.

    > moxgo serve


the builtin cli makes it easy to play with popular classification models

    > moxgo pull birdnet
    > moxgo run birdnet --file what_bird_is_this.mp3

    > moxgo pull yolo11n
    > moxgo run yolo11n --file is_this_a_dog.png


live detection
--------------

`--display` runs a model on live video and shows it with boxes, labels and masks
drawn in. Video goes from the camera to the window at the camera's rate;
detection takes the newest frame whenever it is free and never holds video up.

    > moxgo run yolo11n --camera 0 --camera-res 1280x720 --display
    > moxgo run yolo11n --stream rtsp://192.168.1.10/stream --display --stats
    > moxgo run yolo11n --onvif http://192.168.1.10:8000/onvif/device_service --onvif-profile sub --display
    > moxgo cameras --onvif

camera credentials come from the URL, or better from `MOXGO_CAMERA_USER` and
`MOXGO_CAMERA_PASSWORD`, and never appear in output. `--stats` prints per-stage
timings; `--duration 30s` stops by itself.


adding custom models
--------------------

you can add your own ONNX models without publishing them to the registry.
place the model files in `~/.moxgo/models/<name>/` with a `manifest.json` and a `model.onnx`.

    ~/.moxgo/models/my-model/
    ├── manifest.json
    ├── model.onnx
    └── labels.csv        (optional)

the manifest describes inputs, outputs, and runtime options:

```json
{
  "inputs": {
    "images": {
      "type": "image",
      "width": 640,
      "height": 640
    }
  },
  "outputs": {
    "output0": {
      "labels": "coco80"
    }
  }
}
```
