## Designing and Implementing Custom Backbone Networks for Faster R-CNN:
Google Colab file: https://colab.research.google.com/drive/1Z3g5OOOuPlrM1T8MLHf8dJPGz7ZY4EMV?usp=sharing 

This project designs and implements two custom ‘backbone’ CNNs for the PyTorch Faster R-CNN Object Detection model. This required me to examine and work closely with the PyTorch Faster R-CNN source code to develop a compatible backbone.

I follow the work of The Ocean Cleanup who have recently given attention to discarded plastic in rivers - I wanted to explore using Object Detection models to analyse aerial photographs taken by drone as a way of reducing the need for environmental workers to survey on foot / by small boat. Research published in the the 2024 Environmental Pollution journal found that “Water hyacinths trap between 54% and 77% of surface plastics in the Saigon river” -identification and removal of trapped plastic helps to minimize leaching of microplastics into the ecosystem. Water hyacinth seeds can remain viable for up to 20 years; making this an ongoing problem that needs monitoring. The target users for this project are researchers/volunteers looking to assess the plastic concentration in a river and identify accumulation ‘hot spots’. Automated analysis of drone imagery could facilitate faster cleanup coordination efforts.

Models have been trained on a dataset of drone images capture along the Saigon River. The Models detect plastic and water hyacinth ; overlapping plastic and water hyacinth bounding boxes identify trapped plastic. <br/>
Dataset source: https://data.4tu.nl/articles/_/21648152/1 <br/>
(Images were annotated with bounding boxes using Roboflow)

The aerial imagery presents the ‘small object detection problem’ ; items of plastic are smaller compared to the dimensions of the image. This project explores the use of feature pyramid networks (FPNs), channel and spatial attention (CBAM) and biomimetic convolutional blocks (inspired by eagle eyes) to address the small object detection problem. 

Custom Backbone 1: A combination of residual connections, convolutional block attention modules (CBAM) and a top-down feature pyramid network (FPN).

**Custom Backbone 1 Architecture:** <br/>
<img width="800" alt="image" src="https://github.com/user-attachments/assets/6c100542-2664-421c-817f-8402e410f7f8" />

Custom Backbone 2: A biomimetic backbone that mimics the visual pathways in eagle ways. This backbone incorporates the Shallow and Deep Fovea Modules of EVMNet proposed by [Chen et Lin](https://www.sciencedirect.com/science/article/abs/pii/S1051200424005815). The Fovea modules are combined with Convolutional block attention modules and a top-down feature pyramid network.

**Custom Backbone 2 Architecture:**<br/>
<img width="700" alt="image" src="https://github.com/user-attachments/assets/d01678c1-613d-4300-a3e6-787ac8893106" /> <br/>
<img width="800" alt="image" src="https://github.com/user-attachments/assets/14f73494-993c-474f-b015-817b205d65f5" /> <br/>
<img width="800" alt="image" src="https://github.com/user-attachments/assets/3fb9ab40-a47b-4aef-9c6a-e5047d739d12" /> <br/>


The biomimetic backbone outperformed custom backbone 1; this backbone has been placed in a demo notebook for inference sampling.
Model weights ‘.pth’ file: https://drive.google.com/file/d/1-doWl6zGmP-aPTNoFe-_q2i1w0J16eWF/view?usp=drive_link

A sample prediction from the Faster R-CNN + biomimetic backbone can be seen below: <br/>
<img width="900" alt="image" src="https://github.com/user-attachments/assets/bb93c2e3-5146-4a7d-809c-b89b2ae1f660" />

**Model Results:**<br/>
I fine-tuned YOLOv12 and ResNet50 Faster R-CNN on the chosen dataset to define a benchmark:
<img width="650" alt="image" src="https://github.com/user-attachments/assets/51db980f-9f95-4672-82d4-fff9a424bbf7"/>
