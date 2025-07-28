from .trainer import *
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
import cv2
from detectron2.utils.visualizer import ColorMode
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask
from sahi.utils.import_utils import check_requirements
from typing import List, Optional

class CelloTypePredictor:
    def __init__(self, model_path, confidence_thresh=0.3, max_det=1000, device='cuda', config_path='cellotype/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'):
        self.model_path = model_path
        self.confidence_thresh = confidence_thresh
        self.config_path = config_path
        self.max_det = max_det
        self.device = device
        self.setup()
        
        self.predictor = DefaultPredictor(self.cfg)

    def setup(self):
        """
        Create configs and perform basic setups.
        """
        self.cfg = get_cfg()

        # for poly lr schedule
        add_deeplab_config(self.cfg)
        add_maskdino_config(self.cfg)
        self.cfg.merge_from_file(self.config_path)

        self.cfg.MODEL.IN_CHANS = 3
        self.cfg.SOLVER.AMP.ENABLED = False
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.TEST.DETECTIONS_PER_IMAGE = self.max_det
        self.cfg.MODEL.DEVICE = self.device

        self.cfg.freeze()
        # default_setup(self.cfg, args)
        # setup_logger(output=self.cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")

    def predict(self, image):
        """
        Predict cell types from an image.
        Args:
            image: np.ndarray, shape (H, W, 3)
        """
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        confident_detections = instances[instances.scores > self.confidence_thresh]

        rst = []
        mask_array = confident_detections.pred_masks.numpy().copy()
        num_instances = mask_array.shape[0]
        output = np.zeros(mask_array.shape[1:])

        for i in range(num_instances):
            output[mask_array[i,:,:]==True] = i+1

        output = output.astype(int)
        
        return output
    

class CelloTypeAnnoter:
    def __init__(self, model_path, channels=92, confidence_thresh=0.3, max_det=1000, device='cuda', config_path='configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'):
        self.model_path = model_path
        self.channels = channels
        self.confidence_thresh = confidence_thresh
        self.config_path = config_path
        self.max_det = max_det
        self.device = device
        self.setup()
        
        self.model = Trainer.build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)


    def setup(self):
        """
        Create configs and perform basic setups.
        """
        self.cfg = get_cfg()

        # for poly lr schedule
        add_deeplab_config(self.cfg)
        add_maskdino_config(self.cfg)
        self.cfg.merge_from_file(self.config_path)

        self.cfg.MODEL.IN_CHANS = self.channels
        self.cfg.SOLVER.AMP.ENABLED = False
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.TEST.DETECTIONS_PER_IMAGE = self.max_det
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.MODEL.PIXEL_MEAN = [128 for _ in range(92)]
        self.cfg.MODEL.PIXEL_STD = [11 for _ in range(92)]

        self.cfg.freeze()

    def predict(self, image):
        """
        Annotate cell types on an image.
        Args:
            image: np.ndarray, shape (H, W, N_CHANS)
        """
        height, width = image.shape[:2]
        im = np.transpose(image, (2, 0, 1))
        im = torch.as_tensor(im.astype("float32")).to(self.device)
        inputs = {"image": im, "height": height, "width": width}
        outputs = self.model([inputs])[0]
        instances = outputs["instances"].to("cpu")
        confident_detections = instances[instances.scores > self.confidence_thresh]

        return confident_detections
    

class Detectron2DetectionModel(DetectionModel):
    def check_dependencies(self):
        check_requirements(["torch", "detectron2"])

    def __init__(self, model_path, channels=3, image_size=512, confidence_threshold=0.3, max_det=1000, device='cuda', config_path='cellotype/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'):
        self.model_path = model_path
        self.channels = channels
        self.confidence_threshold = confidence_threshold
        self.config_path = config_path
        self.max_det = max_det
        self.device = device
        self.category_mapping = None
        self.category_remapping = None
        self.image_size = image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None
        self.load_model()

    def load_model(self):
        # data_dir = '/mnt/isilon/tan_lab_imaging/Analysis/Minxing/data/tissuenet_1.0'

        # for d in ["train","val", "test"]:
        #     DatasetCatalog.register("cell_" + d, lambda d=d: np.load(os.path.join(data_dir, 'dataset_dicts_cell_{}.npy'.format(d)), allow_pickle=True))
        #     MetadataCatalog.get("cell_" + d).set(thing_classes=["cell"])
        
        args = default_argument_parser().parse_args()
        args.resume = True
        args.config_file = self.config_path

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.MODEL.IN_CHANS = self.channels
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.MODEL.DEVICE = self.device
        cfg.freeze()
        default_setup(cfg, args)

        model = DefaultPredictor(cfg)

        self.model = model
        self.category_mapping = {'0': "cell"}
        self.category_names = ["cell"]

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        if isinstance(image, np.ndarray) and self.model.input_format == "BGR":
            # convert RGB image to BGR format
            image = image[:, :, ::-1]

        prediction_result = self.model(image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        num_categories = len(self.category_mapping)
        return num_categories

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """

        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        # detectron2 DefaultPredictor supports single image
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        # parse boxes, masks, scores, category_ids from predictions
        boxes = original_predictions["instances"].pred_boxes.tensor
        scores = original_predictions["instances"].scores
        category_ids = original_predictions["instances"].pred_classes

        # check if predictions contain mask
        try:
            masks = original_predictions["instances"].pred_masks
        except AttributeError:
            masks = None

        # filter predictions with low confidence
        high_confidence_mask = scores >= self.confidence_threshold
        boxes = boxes[high_confidence_mask]
        scores = scores[high_confidence_mask]
        category_ids = category_ids[high_confidence_mask]
        if masks is not None:
            masks = masks[high_confidence_mask]

        if masks is not None:
            object_prediction_list = [
                ObjectPrediction(
                    bbox=box.tolist() if mask is None else None,
                    bool_mask=mask.detach().cpu().numpy() if mask is not None else None,
                    category_id=category_id.item(),
                    category_name=self.category_mapping[str(category_id.item())],
                    shift_amount=shift_amount,
                    score=score.item(),
                    full_shape=full_shape,
                )
                for box, score, category_id, mask in zip(boxes, scores, category_ids, masks)
                if mask is None or get_bbox_from_bool_mask(mask.detach().cpu().numpy()) is not None
            ]
        else:
            object_prediction_list = [
                ObjectPrediction(
                    bbox=box.tolist(),
                    bool_mask=None,
                    category_id=category_id.item(),
                    category_name=self.category_mapping[str(category_id.item())],
                    shift_amount=shift_amount,
                    score=score.item(),
                    full_shape=full_shape,
                )
                for box, score, category_id in zip(boxes, scores, category_ids)
            ]

        # detectron2 DefaultPredictor supports single image
        object_prediction_list_per_image = [object_prediction_list]

        self._object_prediction_list_per_image = object_prediction_list_per_image
