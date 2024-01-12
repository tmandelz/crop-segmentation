import numpy as np
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from strenum import StrEnum
import torch
from torcheval.metrics.functional import multiclass_f1_score
import seaborn as sns


class Label(StrEnum):
    """
    Enumeration String for our labels
    """
    Label_0 = "Unknown"
    Label_1 = "Field crops"
    Label_2 = "Forest"
    Label_3 = "Grassland"
    Label_4 = "Orchards"
    Label_5 = "Special crops"



class Evaluation:
    def __init__(self, data_classes: list = [label.value for label in Label]) -> None:
        """
        :param list data_classes: list of true labels (convert from int to str)
        """
        self.classes = data_classes

    def per_batch(self, index_batch: int, epoch: int, loss_batch: float, loss_val: float = None) -> None:
        """
        Logs the loss of a batch
        :param int index_batch: index of the batch to log (step)
        :param int epoch: index of the epoch to log
        :param float loss_batch: loss of the batch for the trainset
        :param float loss_val: loss of the batch for the validationset, is not logged every epoch, defaults to None
        """
        
        wandb.log({"index_batch": index_batch,
                  "epoch": epoch, "loss batch": loss_batch, "loss batch val": loss_val})

    def per_epoch(
        self,
        epoch: int,
        loss_train: float,
        pred_train: np.array,
        label_train: np.array,
        loss_val: float,
        pred_val: np.array,
        label_val: np.array,
    ) -> None:
        """
        wandb log of different scores
        :param int epoch: index of the epoch to log
        :param float loss_train: log loss of the training
        :param np.array pred_train: prediction of the training
        :param np.array label_train: labels of the training
        :param float loss_val: log loss of the validation
        :param np.array pred_val: prediction of the validation
        :param np.array label_val: labels of the validation
        """
        pred_train = torch.tensor(pred_train).flatten()
        label_train = torch.tensor(label_train).flatten()
        pred_val = torch.tensor(pred_val).flatten()
        label_val = torch.tensor(label_val).flatten()
        
        cm = confusion_matrix(label_val, pred_val, labels=[0, 1, 2, 3, 4, 5])
        cm_f = cm.astype(float)
        cm_sm = cm_f[1:,1:]
        
        overall_acc = np.sum(np.diag(cm_sm)) / np.sum(cm_sm)
        
        recall = np.diag(cm_sm)/(np.sum(cm_sm, axis=1) + 1e-12)
        precision = np.diag(cm_sm)/(np.sum(cm_sm, axis=0) + 1e-12)
        f1_test = (2 * precision * recall) / ((precision + recall) + 1e-12)
        
        f1_train ={f"train f1_score_global": multiclass_f1_score(pred_train, label_train, num_classes=6, average='micro')}
            
        
        log = {"epoch": epoch, "Loss train": loss_train, "Loss val": loss_val, "overall_accuracy": overall_acc, "test f1_score_global": f1_test.mean()}
        wandb.log({**f1_train, **log})

    def per_model(self, label_val, pred_val) -> None:
        """
        wandb log of a confusion matrix and plots of wrong classified animals
        :param np.array label_val: labels of the validation
        :param np.array pred_val: prediction of the validation
        :param pd.dataframe val_data: validation data
        """
        true_label = label_val.flatten()
        true_pred = pred_val.flatten()        
        
        cm = confusion_matrix(true_label, true_pred, labels=[0, 1, 2, 3, 4, 5])
        disp = ConfusionMatrixDisplay(cm, display_labels=self.classes)
        disp.plot()
        plt.xticks(rotation = 45)
        
        
        cm_f = cm.astype(float)
        cl_acc = np.diag(cm_f) / (cm_f.sum(1) + 1e-12)
        
        
        wandb.log(
            {
                "confusion matrix": wandb.Image(plt)
            }
        )
        plt.close()
                
        wandb.log(dict(zip(self.classes,cl_acc)))
        
        
        n_samples = 6
        random_fields = np.random.choice(list(range(0,label_val.shape[0])),size=n_samples,replace=False)        
        data_list = np.vstack((label_val[random_fields],pred_val[random_fields]))
        
        all_unique_numbers = [0, 1, 2, 3, 4, 5]
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_unique_numbers)))
        color_map = {num: colors[i] for i, num in enumerate(sorted(all_unique_numbers))}

        fig, axes = plt.subplots(2, n_samples, figsize=(12, 4))
        axes_flat = axes.flatten()

        for i, data in enumerate(data_list):
            
            used_colors = list(map(color_map.get, np.unique(data)) )
            sns.heatmap(data, cmap=used_colors, cbar=False, ax=axes_flat[i])
            if i < n_samples:
                title_text = "Target"
                number_field = i
            else:
                title_text = "Prediction"
                number_field = i-n_samples
            axes_flat[i].set_title(f'{title_text} field {number_field+1}',fontsize = 8)
            axes_flat[i].set_xticks([])
            axes_flat[i].set_yticks([])
            axes_flat[i].set_xticklabels([])
            axes_flat[i].set_yticklabels([]) 
        fig.legend(title='Crop Classes')
                          
        wandb.log({"Example Fields": wandb.Image(plt)})
        plt.close()