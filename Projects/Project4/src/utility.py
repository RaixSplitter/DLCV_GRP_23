from sklearn.metrics import confusion_matrix, precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_iou(bbox1 : list[float], bbox2 : list[float]) -> float:
    # bbox : [x, y, w, h]    

    #Unpack bboxes
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    #Calculate areas of bboxes
    a1 = w1 * h1
    a2 = w2 * h2

    #Calculate intersection box corners, we denote intersection by z
    zx1, zy1 = max(x1, x2), max(y1, y2) #Bottom lefthand corner of intersection box
    zx2, zy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2) #Top righthand corner of intersection box
    zw, zh = zx2 - zx1, zy2 - zy1 #intersection width and height

    # Calculate intersection area and union area
    za = max(0, zw) * max(0, zh) #intersection area
    ua = a1 + a2 - za #union area, u denotes union

    # Calculate IOU
    iou = za / ua

    return float(iou)

def get_pred_from_confidence(confidence : list[float], threshold : float = 0.5) -> list[bool]:
    y_pred = [True if score >= threshold else False for score in confidence]
    return y_pred

def no_max_supression(bboxes, conf_threshold=0.7, iou_threshold=0.4) -> list[list[float]]:

    #bbox : [x, y, w, h, class, confidence] 
    #bboxes : list[bbox]
    

    bbox_list_threshold = []
    bbox_list_new = []
    
    #Sort and filter bbox
    boxes_sorted = sorted(bboxes, key=lambda x: x[5], reverse=True)
    for bbox in boxes_sorted:
        if bbox[5] > conf_threshold:
            bbox_list_threshold.append(bbox)
    
    #Remove boxes with high IOU
    while len(bbox_list_threshold) > 0:
        current_bbox = bbox_list_threshold.pop(0)
        bbox_list_new.append(current_bbox)
        for bbox in bbox_list_threshold:
            if current_bbox[4] == bbox[4]:
                iou = calculate_iou(current_bbox[:4], bbox[:4])
                if iou > iou_threshold:
                    bbox_list_threshold.remove(bbox)

    return bbox_list_new

#mAP
def mean_average_precision(bbox_true : list[list[float]], bbox_pred : list[list[float]], threshold_iot : float = 0.5, plot = False) -> float:
    #bbox = [x, y, w, h, class, confidence]

    #sort bboxes by confidence
    boxes_sorted = sorted(bbox_pred, key=lambda x: x[5], reverse=True)

    #slice duplicate bbox_true
    bbox_gt = bbox_true[:]

    match = []

    for bbox in boxes_sorted:
        for gt in bbox_gt:
            #compute iou
            iou = calculate_iou(bbox[:4], gt[:4])
            if iou > threshold_iot and bbox[4] == gt[4]:
                #remove bbox from gt
                bbox_gt.remove(gt)

                #append bbox to match
                match.append(True)
                break
        else:
            match.append(False)
        
    #calculate precision and recall
    precision = [sum(match[:i+1]) / (i+1) for i in range(len(match))]
    recall = [sum(match[:i+1]) / len(bbox_gt) for i in range(len(match))]

    #calculate average precision
    average_precision = 0
    for i in range(len(precision)-1):
        average_precision += (recall[i+1] - recall[i]) * precision[i+1]
    average_precision += recall[0] * precision[0]

    if not plot:
        return average_precision, precision, recall


    #plot precision-recall curve
    fig, ax = plt.subplots()
    sns.scatterplot(x = recall, y = precision, marker = '*', s = 200, ax=ax)
    for i in range(len(precision)):
        ax.annotate(i+1, (recall[i], precision[i]))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    plt.show()
    plt.savefig("precision_recall_curve.png")

    return average_precision, precision, recall

def sklearn_map(y_true : list[bool], confidence : list[float], threshold : float = 0.5) -> None:
    y_pred = get_pred_from_confidence(confidence, threshold)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()
    plt.savefig("SKprecision_recall_curve.png")    

def get_confusion_matrix(y_true : list[bool], confidence : list[float], threshold : float = 0.5, plot : bool = False) -> None:
    y_pred = get_pred_from_confidence(confidence, threshold)

    #cm
    cm = confusion_matrix(y_true, y_pred)


    if not plot:
        return cm
    #seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig("confusion_matrix.png")

    return cm

def get_metrics(y_true : list[bool], y_pred : list[bool]) -> dict:
    #if y_pred is confidence scores, convert to bool
    if len(y_pred) > 0 and type(y_pred[0]) == float:
        y_pred = get_pred_from_confidence(y_pred)
    
    metric = {}
    #Calculate metrics
    metric["accuracy"] = accuracy_score(y_true, y_pred)
    metric["precision"] = precision_score(y_true, y_pred)
    metric["recall"] = recall_score(y_true, y_pred)
    metric["f1"] = f1_score(y_true, y_pred)
    metric["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metric["average_precision"] = average_precision_score(y_true, y_pred)
    metric["jaccard"] = jaccard_score(y_true, y_pred)

    return metric



if __name__ == '__main__':
    print("Running utility.py")
    bbox1 = [0, 0, 10, 10]
    bbox2 = [5, 5, 10, 10]
    iou = calculate_iou(bbox1, bbox2)
    print(iou)

    y_pred = [0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.7, 0.6, 0.3, 0.5]
    y_true = [True, False, False, False, True, False, True, False, True, True]
    print(y_true)

    sklearn_map(y_true, y_pred)
    get_confusion_matrix(y_true, y_pred, plot = True)


    
    print("Done running utility.py")
