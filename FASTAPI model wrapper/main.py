from fastapi import FastAPI, File, UploadFile
import torch
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection._utils import overwrite_eps
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from torchvision.utils import draw_bounding_boxes
import torch.nn as nn
from utility_funcs import pred_confidence_filter, find_overlapping_bboxes
import io 
import base64
from PIL import Image
from torchvision import transforms

# define model and load weights 
num_trainable_backbone_layers = 3 # valid range for resnet50 is 0-5 according to PyTorch docs
backbone = resnet50(weights=None, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
backbone = _resnet_fpn_extractor(backbone, num_trainable_backbone_layers)

#use default values for a baseline implementation
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))

# default values
anchor_aspect_ratios = ((0.5, 1.0, 2.0),) * len (anchor_sizes)

anchor_generator = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=anchor_aspect_ratios
)
#use default values for a baseline implementation
roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
                                                output_size=7,
                                                sampling_ratio=2)

from torchvision.models.detection.roi_heads import RoIHeads
custom_roi_heads = RoIHeads(
            box_roi_pool = None,
            box_head = None,
            box_predictor = None,
            # Faster R-CNN training
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            # Faster R-CNN inference
            score_thresh=0.001, #default is 0.05,
            nms_thresh=0,  #default is 0.5,
            detections_per_img=100
        )

custom_model = FasterRCNN(
    backbone=backbone,
    num_classes=3,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

#override the prediction heads to have the correct number of classes
num_classes=3 #3 classes +1 to account for background
in_features = custom_model.roi_heads.box_predictor.cls_score.in_features
custom_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# load fine tuned weights 
weights_path = './model_weights/resnet_50_weights.pth'
custom_model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device('cpu')))

app = FastAPI()

@app.get('/')
def health_check():
    return {'health_check': 'OK'}

@app.get('/info')
def info():
    return {'name': 'Plastic-Finder', 
    'description':'an ML model that analyses drone imagery to identify accumulation hot spots'
    }


@app.post('/analyse_image')
async def get_predictions(file: UploadFile): # = File(...)
    # convert image to tensor form
    image = await file.read()
    image = Image.open(io.BytesIO(image))
    # debugging
    # image.show()
    image_transform = transforms.ToTensor()  #cv library reads as a numpy array, needs to be a pytorch tensor to be compatible
    image = image_transform(image)
    
    # set model to eval 
    custom_model.eval()

    # get model result 
    results = custom_model([image])
    results = results[0]
    
    confidence_threshold = 0.5
    filtered_results = pred_confidence_filter(results, confidence_threshold)
    
    trapped_plastic = find_overlapping_bboxes(filtered_results)
    if len(trapped_plastic) > 0:
        img_with_predictions = draw_bounding_boxes(image, 
                                                torch.Tensor(trapped_plastic), 
                                                colors=['orange']*len(trapped_plastic),
                                                labels=['plastic']*len(trapped_plastic),
                                                width=5),
        # base64 encode ready to send back to client 
        pil_img = transforms.ToPILImage()(img_with_predictions[0]).convert("RGB")
        # reduce image size as base 64 encoding is relatively slow ; minimise image size
        MAX_SIZE = (pil_img.size[0]/2, pil_img.size[1]/2) 
        pil_img.thumbnail(MAX_SIZE) 
        img_bytes = pil_img.tobytes()
        base_64_string = base64.b64encode(img_bytes)

        # debugging to check image is valid
        # image = base64.b64decode(base_64_string, validate=True)
        # stream = io.BytesIO(image)
        # new = Image.frombytes('RGB', (pil_img.size[0], pil_img.size[1]), stream.getvalue())
        # new.show()
        # end of debugging

        return {'results': {
            'number_trapped_plastic': len(trapped_plastic),
            'image': base_64_string,
            
        }}
    
    else:
        return {'results': {
            'number_trapped_plastic': len(trapped_plastic),
        }}
    
# fastapi dev main.py

