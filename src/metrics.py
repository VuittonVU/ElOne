from sklearn.metrics import f1_score, classification_report, confusion_matrix


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def make_report(y_true, y_pred, target_names=None):
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)


def make_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
