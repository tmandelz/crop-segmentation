# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import Dataset
from src.evaluation import Evaluation



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
        rescale: bool = True,
        temporal_sampling=True,
        augment_rate=0.66        
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
        self.temporal_sampling = temporal_sampling
        self.augment_rate = augment_rate
        
    def create_loader(self, eval_mode=False):
        """
        Creates the train/test (80/20) loader for all the models
         
        """ 
        if eval_mode:
            dataset_train = Dataset(self.datadir, 0., 'test', True, 3, self.gt_path, num_channel=4, apply_cloud_masking=False,small_train_set_mode=False,temporal_sampling=self.temporal_sampling, augment_rate=self.augment_rate)
            dataset_test = Dataset(self.datadir, 0., 'test', True, 4, self.gt_path, num_channel=4, apply_cloud_masking=False,small_train_set_mode=False,temporal_sampling=self.temporal_sampling, augment_rate=self.augment_rate)
        else:
            dataset_train = Dataset(self.datadir, 0., 'test', False, 3, self.gt_path, num_channel=4, apply_cloud_masking=False,small_train_set_mode=False,temporal_sampling=self.temporal_sampling, augment_rate=self.augment_rate)
            dataset_test = Dataset(self.datadir, 0., 'test', False, 4, self.gt_path, num_channel=4, apply_cloud_masking=False,small_train_set_mode=False,temporal_sampling=self.temporal_sampling, augment_rate=self.augment_rate)

        self.train_loader = torch.utils.data.DataLoader(dataset_train, self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(dataset_test, self.batch_size, shuffle=False, num_workers=0)
  

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
            settings=wandb.Settings(start_method="thread", _service_wait=300),
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
        lr=int,
        batch_size = int,
        loss_module: nn = nn.CrossEntropyLoss(weight=torch.tensor([10**-40, 1, 1, 1, 1, 1]).to(device)),
        weight_decay=0,
        augment_rate = 0.66        
        
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
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # setup a new wandb run
        self.setup_wandb_run(
            run_group,
            fold,
            lr,
            num_epochs,
            model_architecture
            )

        self.batch_size = batch_size
        self.augment_rate = augment_rate
        self.create_loader()
        self.weight_decay=weight_decay
        
        # Overfitting Test for first batch
        if test_model:
            self.train_loader = [next(iter(self.train_loader))]
                 
                
        # training mode
        self.model.train()
        self.model.to(device)
        
        # prepare the model
        optimizer = optim.Adam(self.model.parameters(), lr, self.weight_decay)
        
        # train loop over epochs
        for epoch in tqdm(range(num_epochs), unit="epoch", desc="Epoch-Iteration"):
            loss_train = np.array([])
            label_train_data = np.empty((0, 24, 24))
            pred_train_data = np.empty((0, 24, 24))

            # train loop over batches
            for batch_iter, data in enumerate(self.train_loader, 0):

                # calc gradient
                input, target_glob, target_local_1, target_local_2 = data
                
                
                if torch.cuda.is_available():
                    input = input.to(device)
                    target_local_1 = target_local_1.to(device)
                
                # upscaling UNet
                if self.rescale:
                    
                    up = nn.Upsample(size=(80, 80), mode='bicubic')
                    if input.shape[0]!=1:
                        input = torch.squeeze(input)
                    else:
                        input = input.view(1, 4, 24, 24)
                    
                    input_up = up(input)

                    
                preds = self.model(input_up)                
                
                
                if self.rescale:
                    down = nn.Upsample(size=(24, 24), mode='bicubic')                    
                    preds = down(preds)
                    preds = preds.to(device)               
                
                loss = loss_module(preds, target_local_1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

                
            # wandb per epoch
            pred_val, label_val, preds1 = self.predict(self.model, self.test_loader)
            
            
            label_val_t = torch.tensor(label_val, dtype=torch.long).to(device)
            pred_val_t = torch.tensor(pred_val).to(device)
            preds_1_t = torch.tensor(preds1, dtype=torch.float).to(device)
            
            loss_val = loss_module(preds_1_t, label_val_t)

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
        preds1 = np.empty((0, 6, 24, 24))
        with torch.no_grad():  # Deactivate gradients for the following code
            
            for batch_iter, data in enumerate(data_loader, 0):

                # calc gradient
                input, target_glob, target_local_1, target_local_2 = data
                
                if torch.cuda.is_available():
                    input = input.to(device)
                    target_local_1 = target_local_1.to(device)
                    
                if self.rescale:
                    up = nn.Upsample(size=(80, 80), mode='bicubic')
                    input = torch.squeeze(input)
                    input = up(input)

                preds = model(input)
                preds = preds.to(device)
                
                if self.rescale:
                    down = nn.Upsample(size=(24, 24), mode='bicubic')                    
                    preds = down(preds)
                    preds_1 = nn.functional.softmax(preds, dim=1)
                    preds_2 = torch.argmax(preds_1, dim=1)
                    
                predictions = np.concatenate(
                    (predictions, preds_2.data.cpu().numpy()), axis=0
                )
                
                preds1 = np.concatenate(
                    (preds1, preds_1.data.cpu().numpy()), axis=0
                )

                true_labels = np.concatenate(
                        (true_labels, target_local_1.data.cpu().numpy()), axis=0
                    )
        model.train()
        return predictions, true_labels, preds1
 

    

