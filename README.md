This is my personal ai assistant project.
Runs 100% local on your machine using Ollama

Run ollama pull <model_name> after downloading ollama to download desired model.

Command to check if your pc is fighting for VRAM
nvidia-smi -l 1 --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free --format=csv

Logs of stuff i tried/changed:
 - Replaced entire RVC pipeline (dependency/requirements of hell)
 - 
  
