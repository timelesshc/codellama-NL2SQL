# Text2SQL on CodeLlama 7B
Use LoRA to fine tune codellama-7b-hf to achieve natural language to SQL ability. Deepspeed is also used in this project. 
## Technologies Used
- LoRA
- DeepSpeed

## Run the Code
Download codellama-7b model and save to `codellama-7b-hf` folder.

Run command:
```
deepspeed code_llama_finetune.py
```
## Test Result
Run command:
```
python final_test.py
```
## Result
The final SQL execution accuracy is >70%
