from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import argparse
import json

def eval_keypoints(pred_json, anno_json, sigmas):
    anno = COCO(anno_json)  # init annotations api
    with open(pred_json,'r') as f:
        pred = json.load(f)
    if isinstance(pred, dict):
        pred = pred['annotations']
    pred = anno.loadRes(pred)  # init predictions api
    eval = COCOeval(anno, pred, 'keypoints', sigmas=sigmas) #,
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    #stats = eval.stats
    ##map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    #return stats

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='run_eval.py')
    parser.add_argument('--dataset', type=str, default='soccer', help='soccer or basketball dataset')
    parser.add_argument('--pred', type=str, default='detections/pipeline_soccer_result.json', help='prediction json path with COCO result format')
    parser.add_argument('--anno', type=str, default='annotations/soccer_groundtruth.json', help='groundtruth json path')
    opt = parser.parse_args()


    sigmas = {}
    #                               ear    NECK r_sho/elb/wrist l_sho/elb/wri r_hip/knee/ankle l_hip/kn/ankle nose
    sigmas['soccer'] = np.array([.35, .35, .55, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .26])/10.0
    #                                    ear    nose neck shoulder    elbox     wrist      HAND      hip        knee     ankle      FOOT
    sigmas['basketball'] = np.array( [.35, .35, .26, .35, .79, .79, .72, .72, .62, .62, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .87, .87])/10.0 # sigma in OKS for keypoint detection

    eval_keypoints(opt.pred, opt.anno, sigmas[opt.dataset])
    #with open(opt.pred,'r') as f:
    #    pred = json.load(f)
    #if isinstance(pred, dict):
    #    pred = pred['annotations']
    #with open(opt.pred,'w') as f:
    #    json.dump(pred,f)

