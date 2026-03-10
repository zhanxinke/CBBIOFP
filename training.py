import time 
import torch
import hydra

from tqdm import tqdm
from rich.console import Console
from omegaconf import OmegaConf

from data import *
from data.dataloader import DataLoader

from models import *
from models.CBBIOFP import *
from models.CBBIOFPT import *

from models.nx_xent import NT_Xent
from models.MultiSupCon import MultiSupConLoss
from utils.parameter_out import parameter_out
from models.pretrain import *
from utils.evaluation import *

from src import folders

class Trainer:
    def __init__(self, args):
        self.args = args
        
        self.console = Console()
    
        self.console.log('=>  Initial Settings')
        set_seed(args.basic.seed)
        self.out_path = args.ds.r_savepath.path
        self.console.log(f'[green]    => Train in {args.ds.basic.dataset}')
        self.console.log(f'[red]    => Setting Seed: {args.basic.seed}')

        self.console.log('=>  Initial Models')
        self.loss_fn   = get_loss_fn(args)
        self.model     = CBBIOMFPT(args).cuda()

        para_mb = str(float(count_parameters_in_MB(self.model)))
        self.console.log(f'[red]     => Supernet Parameters: {para_mb} MB')

        self.console.log('=>  Preparing Dataset')
        # if args.ds.basic.task == 'classification':
        #     self.train_dataset, self.test_dataset = load_data(args)
        #     self.console.log(f'[red]    => train_data: {self.train_dataset[0][0]}')
        #     self.console.log(f'[red]    => test_data: {self.test_dataset[0][0]}')
        
        self.train_dataloader, self.test_dataloader = self.load_dataloader()
        self.optimizer = torch.optim.Adam(
                    params       = self.model.parameters(),
                    lr           = args.optimizer.lr,
                    weight_decay = args.optimizer.weight_decay)
        self.console.log('=>  Initial Optimizer && Scheduler')
        
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer, 
            base_lr        = self.args.optimizer.lr, 
            max_lr         = self.args.optimizer.lr * 10, 
            cycle_momentum = False,
            step_size_up   = len(self.train_dataloader) // self.args.ds.train_params.batch)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 20, gamma=0.9, last_epoch=-1)
        
    def load_dataloader(self):
        dataset = folders.Folder()
        tr_data = folders.Dataset(dataset.train_seq, dataset.train_data, dataset.train_label)
        te_data = folders.Dataset(dataset.test_seq, dataset.test_data, dataset.test_label)
        train_data = DataLoader(batch_size=64, istrain=True).get_data(tr_data)
        test_data = DataLoader(batch_size=64, istrain=True).get_data(te_data)
        return train_data, test_data

    def run(self):
        loss_lst = []

        for epoch in range(self.args.ds.train_params.epoch):

            # ================= train =================
            self.model.train()
            loss_sum = 0

            for i, (seq, data, label) in enumerate(self.train_dataloader):
                seq = seq.cuda()
                data = data.cuda()
                label = label.cuda().float()

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_sum += loss.item()

            train_loss = loss_sum / (i + 1)
            self.console.log(f'[red]    => Epoch {epoch} Train Loss: {train_loss:.6f}')
            loss_lst.append(train_loss)

            # ================= test =================
            self.model.eval()
            preds = []
            reals = []

            with torch.no_grad():
                for i, (seq, data, label) in enumerate(self.test_dataloader):
                    seq = seq.cuda()
                    data = data.cuda()
                    label = label.cuda().float()

                    output = self.model(data)
                    prob = torch.sigmoid(output)

                    preds.append(prob.cpu())
                    reals.append(label.cpu())

            preds = torch.cat(preds, dim=0).numpy()
            reals = torch.cat(reals, dim=0).numpy()

            score = evaluate(preds, reals)

            self.console.log(f'[red]    => Epoch {epoch} Test Performance')
            self.console.log(f'[red]    => aiming: {score[0]:.3f}')
            self.console.log(f'[red]    => coverage: {score[1]:.3f}')
            self.console.log(f'[red]    => accuracy: {score[2]:.3f}')
            self.console.log(f'[red]    => absolute_true: {score[3]:.3f}')
            self.console.log(f'[red]    => absolute_false: {score[4]:.3f}\n')

@hydra.main(config_path = 'config', config_name = 'defaults', version_base=None)
def app(args):
    start_time = time.time()
    console = Console()
    OmegaConf.set_struct(args, False)
    
    create_dir(args.ds.r_savepath.path)
    parameter_out(args)
    Trainer(args).run()

    end_time = time.time()  # 记录程序结束运行时间
    console.log(f'[red]    => Completely Cost Time: {(end_time - start_time)/3600} h')




if __name__ == '__main__':
    app()
