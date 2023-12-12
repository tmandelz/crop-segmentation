import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from PIL import Image
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
        self.pred_val = torch.tensor(pred_val).flatten()
        self.label_val = torch.tensor(label_val).flatten()
        
        
        f1_train ={f"train f1_score_global": multiclass_f1_score(pred_train, label_train, num_classes=6, average='micro')}
    
        f1_test = {f"test f1_score_global":multiclass_f1_score(self.pred_val, self.label_val, num_classes=6, average='micro')}
            
            
            
        """f1_train = {
            f"train f1_score von {self.classes[crop]}": f1_score(
                label_train[:, crop], pred_train == crop
            )
            for crop in range(len(self.classes))
        }
        f1_test = {
            f"validation f1_score von {self.classes[animal]}": f1_score(
                label_val[:, animal], pred_val == animal
            )
            for animal in range(len(self.classes))
        }"""
        
        log = {"epoch": epoch, "Loss train": loss_train, "Loss val": loss_val}
        wandb.log({**f1_train, **f1_test, **log})

    def per_model(self, label_val, pred_val) -> None:
        """
        wandb log of a confusion matrix and plots of wrong classified animals
        :param np.array label_val: labels of the validation
        :param np.array pred_val: prediction of the validation
        :param pd.dataframe val_data: validation data
        """
        self.true_label = label_val
        self.true_pred = pred_val
               
        
        labels_agro = np.unique(self.true_label)
        
        wrong_classified = np.where(self.true_label != self.true_pred)[0]

        
        """self.plot_16_pictures(
            pred_val, label_val
        )"""

        """wandb.log(
            {
                "confusion matrix": wandb.sklearn.plot_confusion_matrix(
                    self.label_val, self.pred_val, self.classes
                ),
                "wrong prediction": plt,
            }
        )
        plt.close()"""

        """ 
        data_wrong_class = val_data.iloc[wrong_classified]
        site_most_wrong = data_wrong_class[
            data_wrong_class["site"] == data_wrong_class["site"].value_counts(normalize=True
                                                                              ).index[0]
        ]
        if len(site_most_wrong) < 16:
            self.plot_16_pictures(range(len(site_most_wrong)), data_wrong_class)
        else:
            self.plot_16_pictures(
                np.random.choice(range(len(site_most_wrong)),
                                 size=16, replace=False),
                data_wrong_class,
            ) 

        plt.suptitle("worst site: " +
                     str(data_wrong_class["site"][0]), size=120)
        wandb.log({"Bad site": plt})
        plt.close() """

    def plot_16_pictures(targets,predictions,n_samples=6) -> None:
        """
        plot 16 pcitures
        :param np.array index: index of the chosen observations
        :param pd.DataFrame data: data with the filepath of the images
        """
        random_fields = np.random.choice(list(range(0,targets.shape[0])),size=n_samples,replace=False)
        data_list = np.vstack((targets[random_fields],predictions[random_fields]))
        # Create a colormap that spans the range of unique numbers
        all_unique_numbers = np.unique(data_list)
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_unique_numbers)))
        color_map = {num: colors[i] for i, num in enumerate(sorted(all_unique_numbers))}

        fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))
        axes_flat = axes.flatten()

        for i, data in enumerate(data_list):
            # Create the heatmap for the current data array without a color bar
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

        """
        #TODO: check and maybe rework this
        fig = plt.figure(figsize=(120, 90), dpi=20)
        for n, variable in enumerate(index):
            ax = fig.add_subplot(4, 4, n + 1)
            datapoint = data.iloc[variable]
            ax.imshow(Image.open(
                "./competition_data/" + datapoint["filepath"]))
            ax.set_title(
                f"{self.classes[self.true_pred[variable]]} anstatt {self.classes[self.true_label[variable]]}",
                size=60,
            )"""
