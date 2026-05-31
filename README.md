MOXGO - the ollama for ONNX
============================

get an easy http api for ONNX models on android,windows,linux,mac,ios with native acceleration.

    > moxgo serve


the builtin cli makes it easy to play with popular classification models

    > moxgo pull birdnet
    > moxgo run birdnet --file what_bird_is_this.mp3

    > moxgo pull yolo11n
    > moxgo run yolo11n --file is_this_a_dog.png


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

input types:

- `image` — requires `width` and `height`
- `audio` — requires `sample_rate` and `chunk` (seconds), optional `overlap` (seconds)
- `raw` — no additional fields

output options:

- `labels` — built-in (`coco80`, `coco91`) or path to a CSV file relative to the model directory
- `sigmoid` — threshold for sigmoid activation on output scores

runtime options (top-level in manifest):

- `ep` — execution provider (e.g. `CUDA`, `CoreML`), defaults to auto-detection
- `threads` — number of inference threads

the server picks up models from `~/.moxgo/models/` on startup. after adding a model, restart the server or use `moxgo run my-model` to load it.

