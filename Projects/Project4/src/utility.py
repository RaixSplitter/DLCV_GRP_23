from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_iou(bbox1 : list[float], bbox2 : list[float]):
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

def get_pred_from_confidence(confidence : list[float], threshold : float = 0.5):
    y_pred = [True if score >= threshold else False for score in confidence]
    return y_pred

def no_max_supression(bboxes, conf_threshold=0.7, iou_threshold=0.4):

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

def mean_average_precision(y_true : list[bool], confidence : list[float], threshold : float = 0.5):
    y_pred = get_pred_from_confidence(confidence, threshold)

    precision_scores = []
    recall_scores = []
    for i in range(1, len(y_pred)):
        #Calculate precision and recall
        precision = precision_score(y_true[:i], y_pred[:i])
        recall = recall_score(y_true[:i], y_pred[:i])

        #Append to list
        precision_scores.append(precision)
        recall_scores.append(recall)

    #Calculate average precision
    average_precision = sum(precision_scores) / len(precision_scores)

    #plot precision-recall curve
    fig = plt.figure(figsize=(10,10))
    plt.plot(x = recall_scores, y = precision_scores)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()
    plt.savefig("precision_recall_curve.png")

    return average_precision, precision_scores, recall_scores

def sklearn_map(y_true : list[bool], confidence : list[float], threshold : float = 0.5):
    y_pred = get_pred_from_confidence(confidence, threshold)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()
    plt.savefig("SKprecision_recall_curve.png")    

def get_confusion_matrix(y_true : list[bool], confidence : list[float], threshold : float = 0.5, plot : bool = False):
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


