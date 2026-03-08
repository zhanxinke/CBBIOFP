from rich.console import Console
from rich.table import Table

def parameter_out(args):
        
    table = Table(title="Parameters Settings",width=150)
    table.add_column("Config", justify="center", style="cyan", no_wrap=True)
    table.add_column("Setting", justify="center", style="magenta")
    table.add_column("Parameter", justify="center", style="green")
    
    for key, value in args.items():
        for k, v in value.items():    
            table.add_row(key,str(k),str(v))

    console = Console(record=True)
    console.print(table)
    
     # 保存为txt
    if args.ds.r_savepath:
        with open(args.ds.r_savepath.path+"/parameter.txt", "w", encoding="utf-8") as f:
            f.write(console.export_text())
