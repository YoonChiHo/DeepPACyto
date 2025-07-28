import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from tqdm import tqdm
import cv2
import json
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_cell_dicts(data_y):
    results = []

    for i in tqdm(range(len(data_y))):

        mask = data_y[i]
        for j in np.unique(mask)[1:]:
            mask_j = mask == j
            # rle = mask_util.encode(np.asfortranarray(mask_j.astype(np.uint8)))
            rle =  mask_util.encode(np.array(mask_j[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            result = {
            "image_id": i,
            "category_id": 0,
            # "bbox": boxes[k],
            "score": 0.9,
            "segmentation": rle,
            }

            results.append(result)
    
    return results

class COCOevalMaxDets(COCOeval):
    """
    Modified version of COCOeval for evaluating AP with a custom
    maxDets (by default for COCO, maxDets is 100)
    """

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results given
        a custom value for  max_dets_per_image
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            # Evaluate AP using the custom limit on maximum detections per image
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.55, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, iouThr=0.6, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, iouThr=0.65, maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, iouThr=0.7, maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[7] = _summarize(1, iouThr=0.8, maxDets=self.params.maxDets[2])
            stats[8] = _summarize(1, iouThr=0.85, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(1, iouThr=0.9, maxDets=self.params.maxDets[2])
            stats[10] = _summarize(1, iouThr=0.95, maxDets=self.params.maxDets[2])
            # stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            # stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            # stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

def ap_eval(ann, res):
    cocoGt = COCO()
    cocoDt = cocoGt.loadRes(res)
    cocoGt.dataset = ann
    cocoEval = COCOevalMaxDets(cocoGt, cocoDt, "segm")
    cocoEval.params.maxDets = [100, 500, 1000]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats