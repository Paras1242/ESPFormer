import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,TrainingArguments,Trainer
import h5py
import numpy as np
import pandas as pd
import os
import csv
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from torch.utils.data import Dataset, DataLoader

results_output_dir = "./results/ESPFormer_custom/"
training_data_dir = "./data/training_data"
validation_data_dir = "./data/validation_data"

def get_pred_prob(trainer, valid_dataset):
    logits_true_lables=trainer.predict(valid_dataset)
    predictions, probs = [], []
    logits=torch.tensor(logits_true_lables[0])
    preds = torch.argmax(logits, dim=-1)
    probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    predictions.extend(preds.cpu().numpy())
    true_labels = logits_true_lables[1]
    return probs, predictions, true_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lis = [["01","02","01"],["02","02","02"],["03","02","05"],["04","05","01"],["05","05","02"],["06","05","05"],["07","15","01"],["08","15","02"],["09","15","05"],["10","30","01"],["11","30","02"],["12","30","05"]]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]*(10**6)
        label = self.labels[idx]
        
        return {
            'input_ids': torch.tensor(input_ids
                                      , dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        

def compute_metrics(eval_preds):
    
    logits, labels = eval_preds
    print(f"logits : {logits} , labels {labels} ")
    predictions = np.argmax(logits,axis=1)
    print(f"predictions : {predictions}")
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
    cm = confusion_matrix(labels, predictions)

    TN, FP, FN, TP = cm.ravel()
    acc = (TP+TN)/(TP+FP+FN+TN)

    sensitivity = TP / (TP + FN)

    Specificity = TN / (TN + FP)

    roc_auc = roc_auc_score(labels, probs)
    eval_dic = {"accuracy":acc,"Sensitivity":sensitivity,"Specificity":Specificity,"ROC AUC Score":roc_auc}
   
    return eval_dic

# **kwargs used to pass a variable number of keyword argument to a function

class CustomConfig(PretrainedConfig):
    model_type = "EEGFormer_Classification"
    def __init__(self,
                input_dim = 20,
                embed_dim = 128,
                num_heads = 8,
                num_layers = 2,
                seq_length = 1280,
                num_classes = 2,
                dropout = 0.1,
                **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.dropout = dropout

# by setting config_class = CustomConfig, we inform the transformer library that
# our EEGFormerModel uses CustomConfig as its configuration class
# EEGFormerModel.from_pretrained('path_to_model'), the method needs to know which configuration class to instantiate.
class EEGFormerModel(PreTrainedModel):
    config_class = CustomConfig
    def __init__(self,config):
        super().__init__(config)

        self.embed_dim = config.embed_dim
        self.seq_length = config.seq_length

        self.embeddings = nn.Linear(config.input_dim, config.embed_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        if self.embeddings.bias is not None:
            nn.init.zeros_(self.embeddings.bias)
        self.cls_token = nn.Parameter(torch.zeros(1,1,config.embed_dim)) # it set requires_grad=True automaticaly
        nn.init.normal_(self.cls_token, std=0.02)
        self. positional_embeddings = nn.Parameter(
            torch.zeros(1, config.seq_length+1, config.embed_dim)
        ) # we can also use torch.randn(1,config.seq_length+1, config.embed_dim) and not use nn.init.normal() code
        nn.init.normal_(self. positional_embeddings, std=0.02)

        #Tranformer Encoder
        #encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = config.embed_dim,
            nhead = config.num_heads,
            dim_feedforward = config.embed_dim*4,
            dropout = config.dropout,
            activation ='gelu',
            batch_first = True
            
        )# It automaticaly intialze the parameters

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = config.num_layers
        )

        #classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim//2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim//2, config.num_classes)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        input_ids = Tensor of shape (batch_size, seq_length, input_dims)
        labels = Tensor of shape (batch_size,)
        """
        batch_size = input_ids.size(0)

        #Embeddings
        x = self.embeddings(input_ids)

        #Add CLS token
        cls_tokens = self.cls_token.expand(batch_size,-1,-1)
        x = torch.cat((cls_tokens,x),dim=1)

        # Add positional embeddings
        x = x + self.positional_embeddings

        x = self.transformer_encoder(x)

        cls_output = x[:,0,:]

        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        output = (logits,)
        return ((loss,) + output) if loss is not None else output

for i in lis:

    #

    train_path=training_data_dir + f"/BM{i[0]}_data.hdf5"
    train_labs = training_data_dir + f"/BM{i[0]}_label.csv"
    valid_path = validation_data_dir + f"/BM{i[0]}_data.hdf5"
    valid_labs = validation_data_dir + f"/BM{i[0]}_label.csv"

 
    print('Reading data to verify correct writes ...')
    X_train_read_hdf = h5py.File(train_path,'r')
    X_train_read = X_train_read_hdf['tracings'][:]
    print('Training values array shape:', X_train_read.shape)
    #X_train_read_hdf.close()
    
    y_train_read_csv = pd.read_csv(train_labs, header=None, index_col = None)
    y_train_read = y_train_read_csv.values.squeeze()
    print('Training labels array shape:', y_train_read.shape)       
    
    X_valid_read_hdf = h5py.File(valid_path,'r')
    X_valid_read = X_valid_read_hdf['tracings'][:]     
    print('Validate values array shape:', X_valid_read.shape)
    #X_valid_read_hdf.close()  
    
    y_valid_read_csv = pd.read_csv(valid_labs, header=None, index_col = None)
    y_valid_read = y_valid_read_csv.values.squeeze()
    print('Training labels array shape:', y_valid_read.shape)
    
    print('Verification Complete!')
    
    train_dataset = TimeSeriesDataset(X_train_read, y_train_read)
    val_dataset = TimeSeriesDataset(X_valid_read, y_valid_read)
            
    config = CustomConfig(
        input_dim=20,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        seq_length=1280,
        num_classes=2,
        dropout=0.1
    )
    model = EEGFormerModel(config).to(device)
    batch_size = 32
    model_name = f"BM_{i[0]}" # ed = embed_dim, h = head, l = layer, dr=droupout
    training_args = TrainingArguments(
        output_dir=f"ModelCheckpoints/ESPFormer/{model_name}",
        run_name = model_name,
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=15,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # tokenizer = processor,
    )
    trainer.train()
    probs, pred, true_labels = get_pred_prob(trainer,val_dataset)

    if i[0][0]=="0":
        BM_num = i[0][1]
    else:
        BM_num = i[0]
    
    prob_path = results_output_dir + f"/ESPFormer_custom/BM{BM_num}_probablity.csv"
    true_path = results_output_dir + f"/ESPFormer_custom/BM{BM_num}_labels.csv"
    # wandb.finish()sss
