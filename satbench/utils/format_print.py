import torch


class FormatTable:
    def __init__(self):
        self.reset()

    def update(self, pred, label):
        pred = pred > 0.5
        self.tn += ((label.eq(0) & pred.eq(0))).sum().item()
        self.fp += ((label.eq(0) & pred.eq(1))).sum().item()
        self.fn += ((label.eq(1) & pred.eq(0))).sum().item()
        self.tp += ((label.eq(1) & pred.eq(1))).sum().item()
    
    def reset(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
    
    def print_stats(self):
        formatted_table = '''       
+--------------------+---------------+-----------------+
|                    |  Labeled sat  |  Labeled unsat  |
+--------------------+---------------+-----------------+
| Predicted sat      |{:15d}|{:17d}|
| Predicted unsat    |{:15d}|{:17d}|
+--------------------+---------------+-----------------+
| Sensitivity (TPR)  |{:33.6f}|
+--------------------+---------------+-----------------+
| Specificity (TNR)  |{:33.6f}|
+--------------------+---------------+-----------------+
| Precision (PPV)    |{:33.6f}|
+--------------------+---------------+-----------------+
| F-1 Score          |{:33.6f}|
+--------------------+---------------+-----------------+
| Overall accuracy   |{:33.6f}|
+--------------------+---------------+-----------------+
        '''.format(
            self.tp,
            self.fp,
            self.fn,
            self.tn,
            self.tpr(),
            self.tnr(),
            self.ppv(),
            self.f1(),
            self.accuracy())
        print(formatted_table)

    def tpr(self):
        """
        TPR = TP/P = TP / (TP+FN)
        :return: sensitivity, recall, hit rate, or true positive rate (TPR)
        """
        numerator = self.tp
        denominator = self.tp + self.fn
        if denominator == 0:
            return -1
        return numerator/denominator

    def tnr(self):
        """
        TNR = TN/N = TN / (TN+FP)
        :return: specificity, selectivity or true negative rate (TNR)
        """
        numerator = self.tn
        denominator = self.tn + self.fp
        if denominator == 0:
            return -1
        return numerator / denominator

    def ppv(self):
        """
        PPV = TP / (TP + FP) = 1 - FDR
        :return: precision or positive predictive value (PPV)
        """
        numerator = self.tp
        denominator = self.tp + self.fp
        if denominator == 0:
            return -1
        return numerator / denominator

    def precision(self):
        """
        PPV = TP / (TP + FP) = 1 - FDR
        :return: precision or positive predictive value (PPV)
        """
        return self.ppv()

    def npv(self):
        """
        NPV = TN / (TN + FN) = 1 - FOR
        :return: negative predictive value (NPV)
        """
        numerator = self.tn
        denominator = self.tn + self.fn
        if denominator == 0:
            return -1
        return numerator / denominator

    def f1(self):
        """
        F_1 = 2 * (PPV * TPR) / (PPV + TPR) = 2*TP / (2*TP + FP + FN)
        :return: F1 Score
        """
        numerator = 2 * self.tp
        denominator = 2 * self.tp + self.fp + self.fn
        if denominator == 0:
            return -1
        return numerator / denominator

    def accuracy(self):
        """
        ACC = (TP + TN) / (P + N) = (TP + TN) / (TP + TN + FP + FN)
        :return: Accuracy
        """
        numerator = self.tp + self.tn
        denominator = self.tp + self.tn + self.fp + self.fn
        if denominator == 0:
            return -1
        return numerator / denominator

