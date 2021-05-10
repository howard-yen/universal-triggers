# Attacking Text Generation Models

## Files and Usage

Models used in this project are from https://huggingface.co/models. 

This folder contains the following files: 
+ `create_adv_token.py` creates the triggers using the optimization procedure described in the paper on a specified model.  
+ `sample_from_gpt2.py` creates output samples from a specified text generation model, optionally using the trigger.
+ `Makefile` commands for running `create_adv_token.py` and `sample_from_gpt2.py` with preset arguments. It's highly recommended to run the python programs using the make commands to make your life easier :)

You can easily swap between different models (for example different sizes of the GPT2 model) by changing the argument for `--model_name_or_path`. 

Although we only ran GPT2, DistilGPT2, GPTNeo, and xlnet as part of our experiment, we designed our program to be able to run any model from huggingface of type `AutoModelForCausalLM`. Specifically, we also included CTRL, OpenAI GPT, Transfo XL, and XLM in the program that you can run for yourself! Note that many of these models take up considerable disk space so we suggest you to change the `cache_dir` argument when calling `from_pretrained` to specify where you want to download the models. We didn't run the other models because they require relatively powerful GPUs to run within a reasonable amount of time, so feel free to try them of yourself if you have the computational resources to do so. 


## Future Work

Some future TODOs, feel free to try it out:
+ Try different concepts besides racism, e.g., get GPT-2 to generate fake Tesla stock reports, fake news, articles, sports, technology, hate speech, etc. You can do this by changing the target_texts to have the content you want. This may be better than fine-tuning the model on a particular domain because (1) you do not need a large collection of documents to fine-tune on and (2) you don't need the compute resources to train an extremely large model. Instead, just write a small sample of target outputs (e.g., 20 hand written sentences) and run the attack in a few minutes on one GPU.
+ Use beam search for the optimization
+ Tune the size of the prepended token
+ Optimize trigger tokens on the 1.3B parameters GPTNeo model, CTRL, OpenAI GPT, Transfo XL, and XLM

Things that are left out that we found were not super necessary. Feel free to add these back:
+ Inside the inner loop sample a batch of racist tweets (or whatever content you'd like) and optimize over it.
+ Sample highly frequent n-grams as the "user input" and optimize the prepended token. This allows the attack to be universal over "any input".
