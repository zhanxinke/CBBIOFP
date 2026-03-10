import os
import sys
import hydra
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

@hydra.main(config_path='config', config_name = 'start_train', version_base=None)
def start(config: DictConfig) -> None:
    
    logger.info(f"✅ Start Training {config['basic']['dataset']}..., Current Working directory: {get_original_cwd()}")
    
    # Save Path
    if os.path.exists(config['save']['path']):
        res_save = Path(config['save']['path'])
        Path(res_save).mkdir(parents = True, exist_ok = True)
    else:
        logger.error(f"Save path {config['save']['path']} does not exist")
        sys.exit(1)
    
    cmd =   f"""cd {get_original_cwd()}&&CUDA_VISIBLE_DEVICES={config.basic.gpu} python training.py \
            ds={config['basic']['dataset']}
            """
            
    os.system(cmd)
    

if __name__ == '__main__':
    start()