
# Import Mask R-CNN
from models.mask_rcnn import maskrcnn_resnet50_fpn_v2
from models.faster_rcnn import FastRCNNPredictor
from models.mask_rcnn import MaskRCNNPredictor
def model_initialize(num_classes, device, dtype):

    # Initialize a Mask R-CNN model with pretrained weights
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # Get the number of input features for the classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # Get the numbner of output channels for the Mask Predictor
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    # Replace the box predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=num_classes)
    # Replace the mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=num_classes)

    # Set the model's device and data type
    model.to(device=device, dtype=dtype);
    # Add attributes to store the device and model name for later reference
    return model
