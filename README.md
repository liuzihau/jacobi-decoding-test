# How to use
### Step 1 generate training data
see ```./configs/data_generate_config.json```, we need to prepare data and model for generating training data.    
The sample data for training is **ShareGPT_Vicuna_unfiltered**, please download it and put in "data_path".    
The model we used is **Qwen2.5-0.5B-Instruct** model from Hugging Face, you can change to other model you want but currently the training script is only for Qwen model.    
After all set, run:
```
python generate_data.py
```
### Step 2 train jacobi model
see ```./configs/train_config_local.json``` and put your generated data in the "datapath"    
If you want to use wandb, you can type your api on the "api_key" so that you can see the curve.
run:
```
python train.py
```
### Step 3 evaluate
see ```./configs/inference_config_local.json``` and put your pretrained model, finetuned model in "basepath" and "statepath". Also put the data in "datapath"    
run:
```
python inference.py
```
