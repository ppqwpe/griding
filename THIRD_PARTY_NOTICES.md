# Third-party components

Griding bundles the software and models below. Each keeps its own license; this file lists them so you can check the terms before sharing the app.
Licenses for Python packages are taken from the installed package metadata (`app/lib/*.dist-info`). The FFmpeg license is what the bundled binary reports with `ffmpeg -L`.

## Models

| Model | Source | License |
|---|---|---|
| Head detection, YOLO11s and YOLO11m (`app/model/model.onnx`, `app/model/head_m/model.onnx`) | [deepghs/real_head_detection](https://huggingface.co/deepghs/real_head_detection) | Not stated on the model page. The models are YOLO11 weights trained with [Ultralytics](https://github.com/ultralytics/ultralytics), which is AGPL-3.0, so treat them as AGPL-3.0 unless the authors say otherwise. |
| Face Landmarker (`app/model/face_landmarker.task`) | [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | Apache-2.0 |

## Programs and libraries

| Component | Version | License |
|---|---|---|
| Python (embedded, `app/python`) | 3.12.10 | PSF License |
| FFmpeg (`imageio_ffmpeg/binaries`, gyan.dev essentials build, built with `--enable-gpl --enable-version3`) | 7.1 | GPL-3.0-or-later |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause |
| ONNX Runtime (DirectML) | 1.24.4 | MIT |
| OpenCV (opencv-python-headless) | 5.0.0 | Apache-2.0 |
| MediaPipe | 1.1.1 | Apache-2.0 |
| NumPy | 2.5.4 | BSD-3-Clause (plus 0BSD, MIT, Zlib for bundled parts) |
| Pillow | — | MIT-CMU |
| matplotlib | — | PSF-based matplotlib license |
| protobuf | — | BSD-3-Clause |
| absl-py, flatbuffers, tzdata | — | Apache-2.0 |
| attrs, fonttools, pyparsing, six | — | MIT |
| certifi | — | MPL-2.0 |
| contourpy, cycler, kiwisolver | — | BSD |
| packaging | — | Apache-2.0 or BSD-2-Clause |
| python-dateutil | — | Apache-2.0 / BSD (dual) |

## What this means when you share Griding

- FFmpeg is GPL-3.0. If you hand the release zip to others, the GPL gives them the right to the FFmpeg source. FFmpeg 7.1 source is at [ffmpeg.org](https://ffmpeg.org/download.html), and `ffmpeg -version` prints the exact build options.
- If the head-detection models are AGPL-3.0, sharing the app also means sharing its source. This repository is public, which covers that.
- Griding's own code (`app/facegrid`, `app/ui.html`, `app/tools`, `app/tests`) has no license file yet, so by default all rights are reserved by the author. Add a `LICENSE` file if you want others to reuse it.

This is a summary, not legal advice.
