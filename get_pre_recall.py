import os
import numpy as np


def calculate_tp_fp_fn(predicted, actual, positive_label):
    """
    predicted: 모델이 예측한 라벨 값들의 리스트
    actual: 실제 라벨 값들의 리스트
    positive_label: Positive 클래스의 라벨 값
    """
    tp, fp, fn = 0, 0, 0
    
    for pred, act in zip(predicted, actual):
        if pred == positive_label and act == positive_label:
            tp += 1
        elif pred == positive_label and act != positive_label:
            fp += 1
        elif pred != positive_label and act == positive_label:
            fn += 1
    
    return tp, fp, fn

def calculate_precision(tp, fp):
    return tp / (tp + fp)

def calculate_recall(tp, fn):
    return tp / (tp + fn)

def calculate_ap50(predicted, confidence_scores, ground_truth, iou_threshold=0.5):
    """
    predicted: 모델이 예측한 Bounding Box 정보들의 리스트
    confidence_scores: 모델이 예측한 Bounding Box의 Confidence Score 리스트
    ground_truth: 정답 Bounding Box 정보들의 리스트
    iou_threshold: IoU 임계값
    """
    # Confidence Score 기준 내림차순으로 정렬
    sorted_indices = np.argsort(confidence_scores)[::-1]
    sorted_predicted = [predicted[i] for i in sorted_indices]
    
    # True Positive(TP), False Positive(FP) 개수 초기화
    num_true_positives = 0
    num_false_positives = 0
    
    # Precision과 Recall 값을 저장할 리스트
    precisions = []
    recalls = []
    
    # 각 Ground Truth Bounding Box들이 맞춰졌는지 여부를 저장할 리스트
    is_ground_truth_detected = [False] * len(ground_truth)
    
    for i in range(len(sorted_predicted)):
        # 현재 예측한 Bounding Box와 Confidence Score
        box = sorted_predicted[i]
        confidence = confidence_scores[sorted_indices[i]]
        
        # IoU 값이 임계값 이상인 Ground Truth Bounding Box 찾기
        ious = [calculate_iou(box, gt) for gt in ground_truth]
        max_iou_index = np.argmax(ious)
        max_iou = ious[max_iou_index]
        
        # IoU가 임계값 이상인 경우, 해당 Ground Truth Bounding Box를 맞춘 것으로 처리
        if max_iou >= iou_threshold and not is_ground_truth_detected[max_iou_index]:
            num_true_positives += 1
            is_ground_truth_detected[max_iou_index] = True
        else:
            num_false_positives += 1
        
        # Precision과 Recall 값을 계산하여 저장
        precision = num_true_positives / (num_true_positives + num_false_positives)
        recall = num_true_positives / len(ground_truth)
        precisions.append(precision)
        recalls.append(recall)
    
    # Precision-Recall 곡선의 Area Under Curve(AUC) 계산
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]
    # Precision-Recall 곡선의 Area Under Curve(AUC) 계산
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]
    # AP 값 계산
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0

    return ap

if __name__ == '__main__':
    # 예시 데이터
    tp = 80
    fp = 20
    fn = 30

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)

    print("Precision:", precision)
    print("Recall:", recall)
