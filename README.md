# Positivity-Sentence-Transformation
The Positivity Sentence Transformation is an innovative initiative aimed at reframing sentences to induce a positive perspective without altering their original meaning. By strategically transforming sentence structures and word choices, we can harness the power of positivity and uplift individuals' outlook on various aspects of life.

This project is an implementation of the t5 model mentioned in the paper "Inducing Positive Perspectives with Text Reframing". The dataset used for training the t5 transformer model is called "Positive Psychology Frames".

# How to run the models? 
## Set up the environment
### Windows

Install anaconda, then run

```conda env create --name pos_sent --file=environment_winx64.yml```

```conda activate pos_sent```

### Linux

```conda env create --name pos_sent```

```conda activate pos_sent```

```pip install -r requirements.txt```


## Train/Download the models
You can train your own models using the ```train.py``` file or you can download the trained models from https://drive.google.com/drive/folders/1zt1PYDNTx7660sUJAFOksW6IKhzHjBRV?usp=sharing. If you download the models then put all the models in a ```models``` folder. If you train your own model, make sure to give a proper path where the models should be saved in the train.py file. The models can be tested using ```test.py``` file and it's output is generated in the ```output``` folder. 

## Run the app
```python app.py```

## Using docker

On the docker command line you can run

```docker run -p 5000:5000 nsd97/induce_pos```


In either case, the app should be running on ```http://127.0.0.1:5000/```.
