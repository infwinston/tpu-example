# Overview

This repo provides a minimal example for using [SkyPilot](https://skypilot.readthedocs.io/en/latest/index.html) on TPUs and TPU Pods that trains a ResNet on CIFAR-10.

To run on TPUs:
`sky launch sky.yaml -c example`

To run on a TPU Pod:
`sky launch sky_pod.yaml -c example-pod`
