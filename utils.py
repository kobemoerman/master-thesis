import numpy as np

def calculate_ap(precision, recall):
    """Calculate Average Precision (AP) given precision and recall arrays."""
    # Append 0 and 1 to precision and recall arrays to simplify calculations
    precision = np.concatenate(([0], precision, [1]))
    recall = np.concatenate(([0], recall, [0]))
    # Calculate the area under the precision-recall curve
    ap = np.sum(np.diff(recall) * precision[:-1])
    return ap

def calculate_map(joint_pred, joint_gt, threshold=10):
    """Calculate Mean Average Precision (MAP) at a given threshold for each joint label."""
    num_samples, num_joints = joint_pred.shape[:2]
    ap_scores = np.zeros(num_joints)
    for i in range(num_joints):
        # Extract x, y and z coordinates of predicted and ground truth joint locations
        pred_x, pred_y, pred_z = joint_pred[:, i, 0], joint_pred[:, i, 1], joint_pred[:, i, 2]
        gt_x, gt_y, gt_z = joint_gt[:, i, 0], joint_gt[:, i, 1], joint_gt[:, i, 2]
        # Calculate Euclidean distance between predicted and ground truth joint locations
        distances = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2 + (pred_z - gt_z)**2)
        # Convert the threshold from cm to m
        threshold_m = threshold / 100
        # Calculate precision and recall at the given threshold
        true_positives = distances <= threshold_m
        false_positives = distances > threshold_m
        true_positives_cumulative = np.cumsum(true_positives)
        false_positives_cumulative = np.cumsum(false_positives)
        precision = true_positives_cumulative / (true_positives_cumulative + false_positives_cumulative)
        recall = true_positives_cumulative / np.sum(true_positives)
        # Calculate Average Precision (AP) for the current joint label
        ap_scores[i] = calculate_ap(precision, recall)
    # Calculate Mean Average Precision (MAP) across all joint labels
    map_score = np.mean(ap_scores)
    return map_score, ap_scores