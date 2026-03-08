import time 
import torch
import hydra

from tqdm import tqdm
from rich.console import Console
from omegaconf import OmegaConf

from data import *
from models import *
from models.CBBIOFP import *
from models.nx_xent import NT_Xent
from models.MultiSupCon import MultiSupConLoss
from utils.parameter_out import parameter_out
from models.pretrain import *

class Trainer:
    def __init__(self, args):
        self.args = args
        
        self.console = Console()
        
        set_seed(args.basic.seed)
        self.out_path = args.ds.r_savepath.path
    
        self.console.log('=>  Initial Settings')
        if args.ds.basic.task == 'pretrain':
            self.console.log(f'[green]    => Pretrain in {args.ds.basic.dataset}')
        else:
            self.console.log(f'[green]    => Train in {args.ds.basic.dataset}')
        self.console.log(f'[red]    => Setting Seed: {args.basic.seed}')

        
        self.console.log('=>  Initial Models')
        self.metric    = load_metric(args)
        self.loss_fn   = get_loss_fn(args)
        self.pretrain_model     = CBBIOMFP(args).cuda()   
        para_mb = str(float(count_parameters_in_MB(self.pretrain_model)))
        self.console.log(f'[red]     => Supernet Parameters: {para_mb} MB')

        self.console.log('=>  Preparing Dataset')
        if args.ds.basic.task == 'pretrain':
            self.train_dataset, self.val_dataset = load_data(args)
            self.console.log(f'train_data: {self.train_dataset[0][0]}')
            self.console.log(f'val_data: {self.val_dataset[0][0]}')
            self.optimizer = torch.optim.Adam(
            params       = self.pretrain_model.parameters(),
            lr           = args.optimizer.lr,
            weight_decay = args.optimizer.weight_decay)
            
        elif args.ds.basic.task == 'Feature' or args.ds.basic.task == 'classification':
            self.train_dataset, self.test_dataset = load_data(args)
            self.console.log(f'train_data: {self.train_dataset[0][0]}')
            self.console.log(f'train_label: {self.train_dataset[0][1]}')
        
        
        self.train_dataloader, self.test_dataloader = self.load_dataloader()
        
        self.console.log('=>  Initial Optimizer && Scheduler')
        
        self.optimizer = torch.optim.Adam(
            params       = self.model.parameters(),
            lr           = args.optimizer.lr,
            weight_decay = args.optimizer.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer, 
            base_lr        = self.args.optimizer.lr, 
            max_lr         = self.args.optimizer.lr * 10, 
            cycle_momentum = False,
            step_size_up   = len(self.train_dataset) // self.args.ds.train_params.batch)
    
    def load_dataloader(self):
        if self.args.ds.basic.task == 'pretrain':
            self.train_dataloader = torch.utils.data.DataLoader(
                dataset     = self.train_dataset,
                batch_size  = self.args.ds.train_params.batch,
                shuffle     = True,
                num_workers = self.args.basic.nb_workers)
            
            self.val_dataloader = torch.utils.data.DataLoader(
                dataset     = self.val_dataset,
                batch_size  = self.args.ds.train_params.batch,
                shuffle     = False,
                num_workers = self.args.basic.nb_workers)
            
        elif self.args.ds.basic.task == 'Feature':
            self.train_dataloader = torch.utils.data.DataLoader(
                dataset     = self.train_dataset,
                batch_size  = self.args.ds.train_params.batch,
                shuffle     = False,
                num_workers = self.args.basic.nb_workers)
            
            self.test_dataloader = torch.utils.data.DataLoader(
                dataset     = self.test_dataset,
                batch_size  = self.args.ds.train_params.batch,
                shuffle     = False,
                num_workers = self.args.basic.nb_workers)
            
        elif self.args.ds.basic.task == 'classification':
            self.train_dataloader = torch.utils.data.DataLoader(
                dataset     = self.train_dataset,
                batch_size  = self.args.ds.train_params.batch,
                shuffle     = True,
                num_workers = self.args.basic.nb_workers)
            
            self.test_dataloader = torch.utils.data.DataLoader(
                dataset     = self.test_dataset,
                batch_size  = self.args.ds.train_params.batch,
                shuffle     = False,
                num_workers = self.args.basic.nb_workers)
            
        return self.train_dataloader, self.test_dataloader
        
        
    def run(self):
        if self.args.ds.basic.task == 'pretrain':
            best_val = float("inf")
            with open(f"{self.out_path}/loss_pretrain.csv", "w") as f:
                f.write("epoch,train_loss,val_loss\n")
            
            for epoch in range(self.args.ds.train_params.epoch):
                train_avg_loss = pretrain(self, epoch, self.pretrain_model, self.train_dataloader, self.optimizer)
                val_avg_loss   = pretrain_val(self, epoch, self.pretrain_model, self.val_dataloader)
                # save best on val
                if val_avg_loss < best_val - 1e-6:
                    best_val = val_avg_loss
                    torch.save(self.pretrain_model.state_dict(), f'{self.out_path}/best_model_state_dict.pt')
                    torch.save(self.pretrain_model, f'{self.out_path}/best_model.pt')
                    
                with open(f"{self.out_path}/loss_pretrain.csv", "a") as f:
                        f.write(f"{epoch},{train_avg_loss},{val_avg_loss}\n")    
                        
            self.console.log(f'[red]    => Pretrain Complete')
            

        elif self.args.ds.basic.task == 'Feature':
            self.pretrain_model.load_state_dict(torch.load(f'{self.args.ds.basic.pretrain_path}'))
            self.pretrain_model.eval()
            feature_graph = torch.Tensor()
            train_bar = tqdm(self.train_dataloader)
            for tem in train_bar:
                peptide, _ = tem[0].cuda(), tem[1].cuda()
                embedding, _, _, _, _ = self.pretrain_model(peptide)
                feature_graph = torch.cat((feature_graph, torch.Tensor(embedding.to('cpu').data.numpy())), 0)

            test_bar = tqdm(self.test_dataloader)
            for tem in test_bar:
                peptide, _ = tem[0].cuda(), tem[1].cuda()
                embedding, _, _, _, _ = self.pretrain_model(peptide)
                feature_graph = torch.cat((feature_graph, torch.Tensor(embedding.to('cpu').data.numpy())), 0)
            
            torch.save(feature_graph, f"{self.out_path}/train_feature_graph.pt")
            torch.save(feature_graph, f"{self.out_path}/test_feature_graph.pt")
            self.console.log(f'train_feature_graph: {feature_graph.shape}')
            self.console.log(f'test_feature_graph: {feature_graph.shape}')
            self.console.log(f'train_feature_graph: {feature_graph[0]}')
            self.console.log(f'test_feature_graph: {feature_graph[0]}')
            
            self.console.log(f'[red]    => Feature Extract Complete')
        
        elif self.args.ds.basic.task == 'classification':
            
            
            self.console.log(f'[red]    => Classification Complete')


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
