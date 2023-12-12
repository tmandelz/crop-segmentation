# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import Dataset
from src.evaluation import Evaluation
from collections import namedtuple


# %%
def _set_seed(seed: int = 42):
    """
    Function to set the seed for the gpu and the cpu
    private method, should not be changed
    :param int seed: DON'T CHANGE
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class DeepModel_Trainer:
    def __init__(
        self,
        datadir: str,
        gt_path: str,
        model: nn.Module,
        device: torch.device = device,
        random_cv_seeds: list = [42, 43, 44, 45, 46],
        rescale: bool = True        
        ) -> None:
        """
        Load the train/test/val data.
        :param str run_group: Name of group.
        :param datadir: directory of the hd5 file
        :param gt_path: directory of the labels.csv
        :param nn.Module model: pytorch deep learning module
        :param torch.device device: used device for training
        :param list random_cv_seeds: what random seeds to use
        :param rescale: if images have to be rescaled for UNet
        """
        self.random_cv_seeds = random_cv_seeds
        _set_seed()
        self.model = model
        self.device = device
        self.evaluation = Evaluation()
        self.datadir = datadir
        self.gt_path = gt_path
        self.rescale = rescale
        
    def create_loader(self):
        """
        Creates the train/test (80/20) loader for all the models
         
        """ 
        dataset_train = Dataset(self.datadir, 0., 'test', False, 3, self.gt_path, num_channel=4, apply_cloud_masking=False,small_train_set_mode=False)
        dataset_test = Dataset(self.datadir, 0., 'test', False, 4, self.gt_path, num_channel=4, apply_cloud_masking=False,small_train_set_mode=False)
                
        self.train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=False, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=0)
  

    def setup_wandb_run(
        self,        
        run_group: str,
        fold: int,
        lr: float,
        num_epochs: int,
        model_architecture: str,
    ):
        """
        Sets a new run up (used for k-fold)
        :param str project_name: Name of the project in wandb.
        :param str run_group: Name of group.
        :param str fold: number of the executing fold
        :param int lr: learning rate of the model
        :param int num_epochs: number of epochs to train
        :param str model_architecture: Modeltype (architectur) of the model
        """
        # init wandb
        self.run = wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project='dlbs_crop-UNet',
            entity="dlbs_crop",
            name=f"{fold}-Fold",
            group=run_group,
            config={
                "learning rate": lr,
                "epochs": num_epochs,
                "model architecture": model_architecture,
            }
        )

    def train_model(
        self,
        run_group:str,
        model_architecture:str,
        fold: str,
        num_epochs: int,
        test_model: bool,
        loss_module: nn = nn.CrossEntropyLoss(),        
        lr=1e-3,        
        validate_batch_loss_each: int = 20,        
        
    ) -> None:
        """
        To train a pytorch model.
        :param str run_group: Name of the run group (kfolds).
        :param str model_architecture: Modeltype (architectur) of the model
        :param int num_epochs: number of epochs to train
        :param nn.CrossEntropyLoss loss_module: Loss used for the competition
        :param int test_model: If true, it only loops over the first train batch and it sets only one fold. -> For the overfitting test.
        :param int cross_validation: If true, creates 5 cross validation folds to loop over, else only one fold is used for training
        :param str project_name: Name of the project in wandb.
        :param int batchsize: batchsize of the training data
        :param int num_workers: number of workers for the data loader (optimize if GPU usage not optimal) -> default 16
        :param int lr: learning rate of the model
        :param int validate_batch_loss_each: defines when to log validation loss on the batch
        :param bool cross_validation_random_seeding: defines whether to use the same seed for each fold or to use different ones

        """
        
        
        # setup a new wandb run
        self.setup_wandb_run(
            run_group,
            fold,
            lr,
            num_epochs,
            model_architecture
            )

               
        # Dataset and Dataloader
        self.create_loader()
        
        # Overfitting Test for first batch
        if test_model:
            self.train_loader = [next(iter(self.train_loader))]
            #self.test_loader = [next(iter(self.test_loader))]
        
        
        # prepare the model
        #model = self.model()
        optimizer = optim.Adam(self.model.parameters(), lr)
        
        # training mode
        self.model.train()
        self.model.to(device)
        
        # train loop over epochs
        #batch_iter = 1
        for epoch in tqdm(range(num_epochs), unit="epoch", desc="Epoch-Iteration"):
            loss_train = np.array([])
            label_train_data = np.empty((0, 24, 24))
            pred_train_data = np.empty((0, 24, 24))

            # train loop over batches
            for batch_iter, data in enumerate(self.train_loader, 0):

                # calc gradient
                input, target_glob, target_local_1, target_local_2 = data
                
                if torch.cuda.is_available():
                    input = input.cuda()
                
                # upscaling UNet
                # upscaling from 24 x 24 x 4 to 572 x 572 x 4
                # https://forum.sentinel-hub.com/t/techniques-to-get-high-resolution-images-of-given-coordinates-through-eo-browser-or-python-package/4161
                if self.rescale:
                    up = nn.Upsample(size=(560, 560), mode='bicubic')
                    input = torch.squeeze(input)
                    input_up = up(input)
                    
                preds = self.model(input_up)
                
                
                if self.rescale:
                    down = nn.Upsample(size=(24, 24), mode='bicubic')                    
                    preds = down(preds)                    
                    
                
                loss = loss_module(preds, target_local_1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val_batch = None
                if batch_iter % validate_batch_loss_each == 0:
                    pred_val, label_val = self.predict(self.model, 
                        self.test_loader)
                loss_val_batch = loss_module(torch.tensor(pred_val),
                                                torch.tensor(label_val))

                self.evaluation.per_batch(
                    batch_iter, epoch, loss, loss_val_batch)

                # data for evaluation                
                label_train_data = np.concatenate(
                    (label_train_data,
                        target_local_1.data.cpu().numpy()), axis=0
                )
                predict_train = torch.argmax(preds, 1).data.cpu().numpy()
                
                pred_train_data = np.concatenate(
                    (pred_train_data, predict_train), axis=0
                )
                loss_train = np.append(loss_train, loss.item())

                # iter next batch
                batch_iter += 1
                
                
            # wandb per epoch
            pred_val, label_val = self.predict(self.model, self.test_loader)
            loss_val = loss_module(torch.tensor(
                pred_val), torch.tensor(label_val))
            
            print(pred_train_data.shape, #24*24*batchsize = 2304
                label_train_data.shape, #24 * 24 *  batchsize = 2304
                pred_val.shape, #2304
                label_val.shape #2304
                )
            self.evaluation.per_epoch(
                epoch,
                loss_train.mean(),
                pred_train_data,
                label_train_data,
                loss_val,
                pred_val,
                label_val,
            )
  

            # wandb per model
            self.evaluation.per_model(
                label_val, pred_val)

            
        self.run.finish()
    
   
 
    def predict(
        self, 
        model: nn.Module,
        data_loader: DataLoader,
    ):
        """
        Prediction for a given model and dataset
        :param nn.Module model: pytorch deep learning module
        :param DataLoader data_loader: data for a prediction
        
        :return: predictions and true labels
        :rtype: np.array, np.array"""

        model.eval()
        predictions = np.empty((0, 24, 24))
        true_labels = np.empty((0, 24, 24))
        with torch.no_grad():  # Deactivate gradients for the following code
            
            for batch_iter, data in enumerate(data_loader, 0):

                # calc gradient
                input, target_glob, target_local_1, target_local_2 = data
                
                if torch.cuda.is_available():
                    input = input.cuda()
                    
                if self.rescale:
                    up = nn.Upsample(size=(560, 560), mode='bicubic')
                    input = torch.squeeze(input)
                    input = up(input)

                preds = model(input)
                
                if self.rescale:
                    down = nn.Upsample(size=(24, 24), mode='bicubic')                    
                    preds = down(preds)
                    preds = nn.functional.softmax(preds, dim=1)
                    preds = torch.argmax(preds, dim=1)
                    
                predictions = np.concatenate(
                    (predictions, preds.data.cpu().numpy()), axis=0
                )

                true_labels = np.concatenate(
                        (true_labels, target_local_1.data.cpu().numpy()), axis=0
                    )
        model.train()
        return predictions, true_labels
 

    
# %%
