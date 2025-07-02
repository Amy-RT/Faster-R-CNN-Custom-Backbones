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
<img width="700" alt="image" src="https://github.com/user-attachments/assets/9949eb53-9924-4431-9b19-db6f169b19fe" />

Custom Backbone 2: A biomimetic backbone that mimics the visual pathways in eagle ways. This backbone incorporates the Shallow and Deep Fovea Modules of EVMNet proposed by [Chen et Lin](https://www.sciencedirect.com/science/article/abs/pii/S1051200424005815). The Fovea modules are combined with Convolutional block attention modules and a top-down feature pyramid network.

**Custom Backbone 2 Architecture:**<br/>
<img width="700" alt="image" src="https://github.com/user-attachments/assets/7afb12bd-6104-4384-8152-00bd9dad77da" /> <br/>
<img width="800" alt="image" src="https://github.com/user-attachments/assets/9555cf19-911f-456d-95c0-ed1ee1970f8d" /> <br/>
<img width="800" alt="image" src="https://github.com/user-attachments/assets/c70c2afa-1db7-48c0-9ebc-e333d6f8be1e" /> <br/>


The biomimetic backbone outperformed custom backbone 1; this backbone has been placed in a demo notebook for inference sampling.
Model weights ‘.pth’ file: https://drive.google.com/file/d/1-doWl6zGmP-aPTNoFe-_q2i1w0J16eWF/view?usp=drive_link

A sample prediction from the Faster R-CNN + biomimetic backbone can be seen below: <br/>
<img width="900" alt="image" src="https://github.com/user-attachments/assets/3a63cfa2-5e8b-4838-98eb-3e18b62d0292" />

**Model Results:**<br/>
I fine-tuned YOLOv12 and ResNet50 Faster R-CNN on the chosen dataset to define a benchmark:
<img width="650" alt="image" src="https://github.com/user-attachments/assets/d4db6c57-82c4-4f60-8593-acf9216e4996" />
