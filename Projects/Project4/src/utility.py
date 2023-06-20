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

def no_max_supression(bboxes, iou_threshold=0.4) -> list[list[float]]:

    #bbox : [[x, y, w, h], class, confidence] 
    #bboxes : list[bbox]

    bbox_list_result = []
    
    #Sort and filter bbox
    boxes_sorted = sorted(bboxes, key=lambda x: x[2], reverse=True)
    
    #Remove boxes with high IOU
    while boxes_sorted:
        current_bbox = boxes_sorted.pop() #get bbox with highest confidence
        bbox_list_result.append(current_bbox)

        for bbox in boxes_sorted[:]:
            if current_bbox[1] == bbox[1]: #if same class
                iou = calculate_iou(current_bbox[0], bbox[0])
                if iou > iou_threshold:
                    boxes_sorted.remove(bbox)

    return bbox_list_result

#mAP
def mean_average_precision(bbox_true : list[list[float]], bbox_pred : list[list[float]], threshold_iot : float = 0.5, plot = False) -> float:
    #bbox = [[x, y, w, h], class, confidence]
    #bbox_gt = [[x,y,w,h], class]

    #sort bboxes by confidence
    boxes_sorted = sorted(bbox_pred, key=lambda x: x[2], reverse=True)

    #slice duplicate bbox_true
    bbox_gt = bbox_true[:]

    match = []

    for bbox in boxes_sorted:
        for gt in bbox_gt:
            #compute iou
            iou = calculate_iou(bbox[0], gt[0])
            if iou > threshold_iot and bbox[1] == gt[1]:
                #remove bbox from gt
                bbox_gt.remove(gt)

                #append bbox to match
                match.append(True)
                break
        else:
            match.append(False)
        
    #calculate precision and recall
    precision = [sum(match[:i+1]) / (i+1) for i in range(len(match))]
    recall = [sum(match[:i+1]) / len(bbox_true) for i in range(len(match))]

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



if __name__ == '__main__':
    print("Running utility.py")
    # bboxes = [
    # [[189, 22, 32, 28], 5, 0.23136208951473236],
    # [[107, 169, 19, 63], 6, 0.4421188235282898],
    # [[109, 169, 17, 66], 6, 0.337564617395401],
    # [[32, 180, 19, 24], 6, 0.24561572074890137],
    # [[108, 173, 17, 59], 6, 0.3550938367843628],
    # [[189, 14, 33, 36], 5, 0.23134583234786987],
    # [[108, 172, 17, 55], 6, 0.20247505605220795],
    # [[9, 149, 23, 26], 5, 0.24931873381137848],
    # [[108, 173, 19, 59], 6, 0.38591429591178894],
    # [[188, 22, 34, 28], 5, 0.1905873566865921],
    # [[108, 173, 18, 59], 6, 0.345272034406662],
    # [[32, 181, 14, 22], 6, 0.24699757993221283],
    # [[108, 172, 17, 60], 6, 0.36766287684440613],
    # [[108, 172, 19, 60], 6, 0.3239368498325348],
    # ]

    # new_bb = no_max_supression(bboxes, iou_threshold=0.4)
    # print(new_bb)


    bbox_true = [[[85.33333333333333, 114.66666666666667, 83.3862433862434, 37.07936507936508], 5]]
    bbox_pred = [[[84, 118, 86, 60], 5, 0.203860804438591]]

    mAP = mean_average_precision(bbox_true, bbox_pred, threshold_iot = 0.5, plot = True)
    
    print("Done running utilit<y.py")
