# Project Description
Use LoRA to fine tune codellama-7b-hf to achieve natural language to SQL ability. Deepspeed is also used in this project. 

# Run the Code
Download codellama-7b model and save to `codellama-7b-hf` folder.

Run below command:
```
deepspeed code_llama_finetune.py
```

# Result
The final SQL execution accuracy is >70%
