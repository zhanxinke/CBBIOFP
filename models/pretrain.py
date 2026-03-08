import torch
from tqdm import tqdm
from models.nx_xent import NT_Xent
from models.MultiSupCon import MultiSupConLoss
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

def pretrain(self, epoch, net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num = 0.0, 0
    progress = Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None),              # ✅ 自动撑满剩余宽度
                TextColumn("{task.percentage:>3.0f}%"), # 可选：显示百分比
                TextColumn("• batch=[red]{task.fields[batch_loss]}"),
                TextColumn("• avg=[red]{task.fields[avg_loss]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                expand=True,                            # ✅ 允许占满整行
                refresh_per_second=10,
            )
    task_id = progress.add_task(
        description=f"TRAIN [{epoch}/{self.args.ds.train_params.epoch}]",
        total=len(data_loader),
        batch_loss="NA",
        avg_loss="NA",
    )

    with progress:
        for tem in data_loader:
            peptide, label = tem[0].cuda(), tem[1].cuda()
            graph1, out_1, org2, out_2, attn_score = net(peptide)

            # ========= 计算 loss =========
            if self.args.ds.loss_fn.loss == 'NT_Xent':
                criterion = NT_Xent(
                    out_1.shape[0],
                    self.args.ds.train_params.temperature,
                    1
                )
                loss = criterion(out_1, out_2)

            elif self.args.ds.loss_fn.loss == 'MultiSupCon':
                features = torch.stack([out_1, out_2], dim=1)
                criterion = MultiSupConLoss()
                loss = criterion(features, label)

            else:
                raise ValueError(f"Unknown loss: {self.args.ds.loss_fn.loss}")

            # ========= 统计 =========
            batch_size = peptide.size(0)
            total_loss += loss.item() * batch_size
            total_num  += batch_size
            avg_loss = total_loss / total_num

            # ========= 反向传播 =========
            train_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            train_optimizer.step()

            # ========= 更新进度条 =========
            progress.update(
                task_id,
                advance=1,
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
            )

    return avg_loss


def pretrain_val(self, epoch, net, data_loader):
    net.eval()
    total_loss, total_num = 0.0, 0

    progress = Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.percentage:>3.0f}%"),
        TextColumn("• val_loss=[red]{task.fields[val_loss]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
        refresh_per_second=10,
    )
    
    task_id = progress.add_task(
        description=f"VAL[{epoch}/{self.args.ds.train_params.epoch}]",
        total=len(data_loader),
        val_loss="NA"
    )
    
    with progress:
        with torch.no_grad():
            for tem in data_loader:
                peptide, label = tem[0].cuda(), tem[1].cuda()
                graph1, out_1, org2, out_2, attn_score = net(peptide)

                if self.args.ds.loss_fn.loss == 'NT_Xent':
                    criterion = NT_Xent(
                        out_1.shape[0],
                        self.args.ds.train_params.temperature,
                        1
                    )
                    loss = criterion(out_1, out_2)

                elif self.args.ds.loss_fn.loss == 'MultiSupCon':
                    criterion = MultiSupConLoss()
                    features = torch.stack([out_1, out_2], dim=1)
                    loss = criterion(features, label)

                batch_size = peptide.size(0)
                total_loss += loss.item() * batch_size
                total_num  += batch_size

                avg_loss = total_loss / total_num

                progress.update(
                    task_id,
                    advance=1,
                    val_loss=f"{avg_loss:.4f}"
                )

    return avg_loss
