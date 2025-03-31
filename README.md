This project designs and implements two custom backbones for the PyTorch Faster R-CNN Object Detection model. 

Models have been trained on a dataset of drone images to detect plastic and water hyacinth (an invasive and rapidly growing riverine plant).
Overlapping plastic and water hyacinth bounding boxes identify plastic that is trapped in water hyacinth. Identification and removal of trapped plastic 
helps to minimize leaching of microplastics into the ecosytem.

Custom Backbone 1: A combination of residual connections, convolutional block attention modules (CBAM) and a top-down feature pyramid network (FPN).

Custom Backbone 2: A biomimetic backbone that mimics the visual pathways in eagle ways. This backbone incorporates the Shallow and Deep Fovea Modules of EVMNet proposed by Chen et Lin. 
                    The Fovea modules are combined with Convolutional block attention modules are and a top-down feature pyramid network.

The biomimetic backbone outperformed custom backbone 1; this backbone has been placed in a demo notebook for inference sampling.

A sample prediction from the Faster R-CNN + biomimetic backbone can be seen below:
![sample_prediction_1](https://github.com/user-attachments/assets/8a47bedd-5b39-452c-946b-1c35a1fb9c68)
