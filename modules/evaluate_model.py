from sklearn import metrics
import matplotlib.pyplot as plt



def evaluate_model(model, features, labels, train = True, plot = False):
    predictions = model.predict(features)
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    roc_auc = metrics.roc_auc_score(labels, predictions)
    auc = metrics.accuracy_score(labels, predictions)
    
    if plot:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    
    if train:
        print('Train accuracy of model: '+str(round(auc, 4)*100))
    else:
        print('Test accuracy of model: '+str(round(auc, 4)*100))
        
    return auc

