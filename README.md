# Face Detect & Align & Recognize

This repo provides python codes for detecting faces in pictures and recognizing detected faces. This repo includes codes from [Insightface](https://github.com/deepinsight/insightface), [DSFD](https://github.com/yxlijun/DSFD.pytorch), and [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets).

The original [Insightface](https://github.com/deepinsight/insightface) repo provides codes using MTCNN and Arcface, and MTCNN provide both detection and alignment function. However, MTCNN's time efficiency is terrible when the input image is very large, or its performance is terrible if we manually shrink the image.

So here this repo provide a solution for efficient large image input.

### Requirements

1. Python2
2. MXNet [1.3.1 recommended]
3. PyTorch [0.4.1 recommended, since 1.0.0 works badly in CUDA9.0 environment.] (optional, if you don't use DSFD, you can manually delete the *DSFD.py* file)
4. NVIDIA GPU (tested on GTX1080, but I guess 6GB memory is OK)

### Face Detection

This repo provides 2 detection method, one is FPN, and another is DSFD. DSFD is pre-trained model borrowed from [DSFD](https://github.com/yxlijun/DSFD.pytorch), and FPN is a coarsely trained version by myself on WIDERFACE.

### Face Alignment

MTCNN consists PNet, RNet, and ONet. We use MTCNN's ONet for face alignment. This repo only provide 1 method for the time being.

### Face Recognize

Arcface-ResNet101 is provided. You can also add other lighter models which use ResNet-50 or MobileNet as backbones.

### Performance

NOTE: MXNet needs warming up! So first several frames may be slow.

60 people in a 4K image(detection using [864, 1536] rescaled image)

- FPN+ONet+Arcface-101 - around 0.5 sec
- DSFD+ONet+Arcface-101 - around 0.6 sec

### Demo

Downloads weights files and put them under ` ./lib/data `:

[DSFD-VGG16](https://www.dropbox.com/s/ty0dtc6jhthd6xa/dsfd_face.pth?dl=0)

[FPN-ResNet101](https://www.dropbox.com/s/fan5gxgtqimm3cv/fpn_widerface-0000.params?dl=0)

[Arcface-ResNet101](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0])

Check the **.yaml* file to set correct absolute path.

```shell
cd ./lib
make clean
make all
cd ..
python demo.py
```

![dets_1](/home/liwen/src/face_detect_recognize/dets_1.jpg)