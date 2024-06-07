from function import *
from model_trainer import TrainModel
import numpy as np
from model import *
from function import plot_info

# 设置随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == '__main__':
    for SAMPLE_METHOD in ["oversample"]:
        for model in [BiLSTM_Conv1d_2, BiLSTM_Conv1d]:
            tensor_direction = f'E:\\deeplearning\\Zhongda\\data_tensor_with_baseline_zhongda.pth'
            train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD)
            model_name = f"Zhongda_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{EPOCH}"
            trainer = TrainModel(model_name, model, train_dataloader, val_dataloader,
                                 criterion_class=FocalLoss(ALPHA_LOSS, GAMMA_LOSS))
            info = trainer.train()
            plot_info(info, model_name)
