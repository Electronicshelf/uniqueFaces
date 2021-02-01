# uniqueFaces
Understating faces a little bit better

## [uniqueFacess](https://):
### PyTorch Implementation
<p align="center"><img width="100%" src="xx.png" /></p>


> [Uche Osahor](https://github.com/Electronicshelf)<sup>1</sup>, 
[xxx]<sup>1</sup>, 
[xxxx]<sup>1</sup>,[xxxx]<sup>1</sup>, <sup>1</sup> [West Virginia University] <br/>
<br/>
[paper]:
<br>

**Abstract:** 


## DATASET
https://

# HINTS
- To use a new sketche style, the sketches must be trained with the model from scratch. Its best to use pictures with a white background or a single background color. This would reduce computational time and place more attention on the facial feataures.


    ### Model configuration.
    > Image default size is set to 256, reperesenting 256x256 pixels
    --image_size, type=int, default=256, help='image resolution'

    ### Training configuration.
     >For Training and Testing, set dataset . presently, its in GUI mode.
    --dataset,
    --batch_size, type=int, default=4, help='mini-batch size'
    --num_iters, type=int, default=10000000, help='number of total iterations for training D'
    
    ### Test configuration.
    > default value represent filename number in data/models directory. currently, 1570000 is the highest value and the best file for testing
    --test_iters, type=int, default=1570000, help='test model from this step'
   
    # Miscellaneous.
    > Select Program mode. default value can be set to 'train', 'test' or 'test_gui' 
    --mode, type=str, default='test_gui', choices=['train', 'test', 'test_gui']
  
    

  STORE SKETCHES FOR TESTING IN THIS FOLDER 
    > cgui_image_dir_skt, type=str, default='data/test_Sketch
  

   > default location for miscellaneous files 
    --log_dir, type=str, default='data/logs
    --model_save_dir, type=str, default='data/models
    --sample_dir, type=str, default='data/samples
    --result_dir, type=str, default='data/results
   
   > Step size defines when to save the sample files while training 
    # Step size.
    --sample_step, type=int, default=500
    
## data_loader_gui.py
	# Dataloader >> All values  are already set as default
    x = "data"
    A = "test_Sketch"

## data_loader.py
	# Dataloader >> All values  are already set as default
	x = "/data"
  

## Citation
If you find this work useful for your research, please cite our [paper](https://):
```



```

## DATA


## License
[MIT](https://choosealicense.com/licenses/mit/)

