# TODO: CAPIRE SE VOGLIAMO IL LOGGER IN QUESTO MODO 
# POTREBBE ESSERE UTILI PER I GRAFICI DA INSERIRE NEL REPORT  
# VOLENDO SI PUO AVERE ANCHE SOLO I LOG TESTUALI SENZA GRAFICI
""" import sys
import os # Aggiungi questo
import wandb
from loguru import logger
from constants import Constants as const

def init_logger_and_wandb(config):
    # 1. Gestione WANDB (Opzionale)
    # Controlla se nella tua config hai messo enable_wandb = True
    if hasattr(config, 'enable_wandb') and config.enable_wandb:
        try:
            wandb.init(
                project=config.model_name if config.model_name else "captain_cook_replica",
                config=config,
                # mode="disabled" # Scommenta questo se vuoi disattivarlo senza cancellare il codice
            )
        except Exception as e:
            print(f"Attenzione: WandB non è partito. Errore: {e}")
    
    # 2. Gestione LOGURU (Essenziale)
    # Assicurati che la cartella 'logging' esista, altrimenti crea errore
    if not os.path.exists("logging"):
        os.makedirs("logging")

    log_config = {
        "handlers": [
            {
                "sink": sys.stdout, # Stampa a video
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
            {
                "sink": "logging/" + "logger_{time}.log", # Salva su file
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
        ],
        "extra": {"user": "usr"},
    }
    logger.configure(**log_config) """
import sys

import wandb
from loguru import logger

from constants import Constants as const


def init_logger_and_wandb(config):
    wandb.init(
        project=config.model_name,
        config=config,
    )
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
            {
                "sink": "logging/" + "logger_{time}.log",
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {module}.{function} | <level>{message}</level> ",
            },
        ],
        "extra": {"user": "usr"},
    }
    logger.configure(**config)
