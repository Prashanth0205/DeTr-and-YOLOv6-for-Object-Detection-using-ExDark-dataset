# COCO
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Metrics
import pandas as pd
import matplotlib.pyplot as plt

# Precision-Recall
def mAP_mAR(enhancement):

    # Paths
    test_path = '../ExDark_COCO/test_set.json'
    results_path = f'../Models/Transformer/lightning_logs/{enhancement}/output/results.json'

    cocoGt = COCO(test_path)

    # Predictions
    results_set = json.load(open(results_path, 'r'))
    cocoDt = cocoGt.loadRes(results_set['annotations'])

    # Initialize COCOEval with ground truth, predictions, and evaluation type
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    # Evaluate, accumulate, and summarize
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    mAP50 = cocoEval.stats[1]
    mAP50_95 = cocoEval.stats[0]
    mAR50_95 = cocoEval.stats[8]

    return mAP50, mAP50_95, mAR50_95

# --------------------------------------------------------------------------------

def training_evol(enhancement, legend=False):
        
    # Paths
    model_path = f"../Models/Transformer/lightning_logs/{enhancement}/output"
        
    # Metrics during training
    df = pd.read_csv(model_path + '/metrics.csv')

    # Training loss
    df_train = df[['epoch', 'training_loss']]
    df_train = df_train[df_train['training_loss'].notna()]
    df_train = df_train.groupby('epoch').tail(1)

    # Validation loss
    df_val = df[['epoch', 'validation/loss']]
    df_val = df_val[df_val['validation/loss'].notna()]
    df_val = df_val.groupby('epoch').tail(1)

    # Plot Loss Evolution
    plt.plot(df_train['epoch'], df_train['training_loss'], label='Training loss')
    plt.plot(df_val['epoch'], df_val['validation/loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(enhancement)
    if legend:
        plt.legend()