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
import pandas as pd


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
        self.model.apply(weights_init_xavier)

        para_mb = str(float(count_parameters_in_MB(self.model)))
        self.console.log(f'[red]     => Supernet Parameters: {para_mb} MB')

        self.console.log('=>  Preparing Dataset')
        self.train_dataloader, self.test_dataloader = self.load_dataloader()
        self.optimizer = torch.optim.Adam(
                    params       = self.model.parameters(),
                    lr           = args.optimizer.lr,
                    weight_decay = args.optimizer.weight_decay)
        self.console.log('=>  Initial Optimizer && Scheduler')
        
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     self.optimizer, 
        #     base_lr        = self.args.optimizer.lr, 
        #     max_lr         = self.args.optimizer.lr * 10, 
        #     cycle_momentum = False,
        #     step_size_up   = len(self.train_dataloader) // self.args.ds.train_params.batch)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 20, gamma=0.9, last_epoch=-1)
        
    def load_dataloader(self):
        dataset = folders.Folder()
        tr_data = folders.Dataset(dataset.train_seq, dataset.train_data, dataset.train_label)
        te_data = folders.Dataset(dataset.test_seq, dataset.test_data, dataset.test_label)
        train_data = DataLoader(batch_size=self.args.ds.train_params.batch, istrain=True).get_data(tr_data)
        test_data = DataLoader(batch_size=self.args.ds.train_params.batch, istrain=True).get_data(te_data)
        return train_data, test_data


    def evaluate_loader(self, model, dataloader, device):
        model.eval()

        preds = []
        reals = []

        with torch.no_grad():
            for seq, data, label in tqdm(dataloader, desc="Evaluating", leave=False):
                seq = seq.to(device, non_blocking=True)
                data = data.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True).float()

                output = model(data)
                prob = torch.sigmoid(output)

                preds.append(prob.cpu())
                reals.append(label.cpu())

        preds = torch.cat(preds, dim=0).numpy()
        reals = torch.cat(reals, dim=0).numpy()

        score = evaluate(preds, reals)
        return score, preds, reals


    def save_result_txt(self, path, title, score):
        with open(path, "w") as f:
            f.write(f"{title}\n")
            f.write(f"aiming: {score[0]:.4f}\n")
            f.write(f"coverage: {score[1]:.4f}\n")
            f.write(f"accuracy: {score[2]:.4f}\n")
            f.write(f"absolute_true: {score[3]:.4f}\n")
            f.write(f"absolute_false: {score[4]:.4f}\n")


    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        os.makedirs(self.out_path, exist_ok=True)

        loss_lst = []
        metric_history = []

        # 这里用 accuracy 作为“越大越好”的 best 指标
        # 你也可以改成 score[3] absolute_true
        best_metric = -1.0
        best_epoch = -1

        for epoch in range(self.args.ds.train_params.epoch):
            # ================= train =================
            self.model.train()
            loss_sum = 0.0

            train_bar = tqdm(
                self.train_dataloader,
                desc=f"Train Epoch {epoch}",
                leave=False
            )

            for i, (seq, data, label) in enumerate(train_bar):
                seq = seq.to(device, non_blocking=True)
                data = data.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True).float()

                self.optimizer.zero_grad()

                output = self.model(data)
                loss = self.loss_fn(output, label)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_sum += loss.item()
                avg_loss = loss_sum / (i + 1)

                train_bar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}"
                })

            train_loss = loss_sum / (i + 1)
            self.console.log(f'[red] => Epoch {epoch} Train Loss: {train_loss:.6f}')
            loss_lst.append(train_loss)

            # ================= val/test for model selection =================
            score, preds, reals = self.evaluate_loader(self.model, self.test_dataloader, device)

            self.console.log(f'[red] => Epoch {epoch} Eval Performance, aiming: {score[0]:.4f}, coverage: {score[1]:.4f}, accuracy: {score[2]:.4f}, absolute_true: {score[3]:.4f}, absolute_false: {score[4]:.4f}')
 

            metric_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "aiming": score[0],
                "coverage": score[1],
                "accuracy": score[2],
                "absolute_true": score[3],
                "absolute_false": score[4],
            })

            # ================= save best =================
            current_metric = score[2]   # 用 accuracy 选 best
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch

                torch.save(self.model.state_dict(), f"{self.out_path}/best_model.pt")

                self.save_result_txt(
                    f"{self.out_path}/best_result.txt",
                    f"Best Model @ Epoch {epoch}",
                    score
                )

                self.console.log(f"[green] => Best model saved at epoch {epoch}, accuracy={best_metric:.4f}")

        # ================= save training history =================
        pd.DataFrame(metric_history).to_csv(
            f"{self.out_path}/train_history.csv",
            index=False
        )

        pd.DataFrame({
            "epoch": list(range(len(loss_lst))),
            "train_loss": loss_lst
        }).to_csv(
            f"{self.out_path}/train_loss.csv",
            index=False
        )

        # ================= load best and final test =================
        self.console.log(f"[cyan] => Loading best model from epoch {best_epoch}")
        self.model.load_state_dict(torch.load(f"{self.out_path}/best_model.pt"))
        self.model.eval()

        final_score, final_preds, final_reals = self.evaluate_loader(
            self.model,
            self.test_dataloader,
            device
        )

        self.console.log(f'[bold green] => Final Test Performance (Best Model), aiming: {final_score[0]:.4f}, coverage: {final_score[1]:.4f}, accuracy: {final_score[2]:.4f}, absolute_true: {final_score[3]:.4f}, absolute_false: {final_score[4]:.4f}')

        self.save_result_txt(
            f"{self.out_path}/final_test_result.txt",
            f"Final Test Performance (Best Model @ Epoch {best_epoch})",
            final_score
        )

        # 如果你还想保存预测值
        torch.save(
            {
                "preds": final_preds,
                "labels": final_reals,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
            },
            f"{self.out_path}/final_test_predictions.pt"
        )

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
