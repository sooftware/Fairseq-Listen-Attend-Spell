# Fairseq-Listen-Attend-Spell
A Fairseq implementation of Listen, Attend and Spell (LAS), an End-to-End ASR framework. LAS architecture described in 
[Listen, Attend and Spell (William Chan etal., 2016)](https://arxiv.org/abs/1508.01211). Fairseq provides several conveniences such as training on multi-GPU, decoding speed, and more.
There is already an [asr example](https://github.com/pytorch/fairseq/tree/master/examples/speech_recognition) in fairseq, but there is no script that implements LAS, so I made it.  
  
## Additional dependencies  
On top of main fairseq dependencies there are couple more additional requirements.  
1. Please follow the instructions to install [torchaudio](https://github.com/pytorch/audio). This is required to compute audio fbank features.  
2. [Sclite](http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm#sclite_name_0) is used to measure WER. Sclite can be downloaded and installed from source from sctk package here. Training and inference doesn't require Sclite dependency.  
3. [sentencepiece](https://github.com/google/sentencepiece) is required in order to create dataset with word-piece targets.  
4. [tensorboard](https://www.tensorflow.org/tensorboard?hl=ko) is required in order to visualize training.
    
## Preparing librispeech data  
```
./examples/speech_recognition/datasets/prepare-librispeech.sh $DIR_TO_SAVE_RAW_DATA $DIR_FOR_PREPROCESSED_DATA  
```

## Training librispeech data  
```
python train.py $DIR_FOR_PREPROCESSED_DATA --save-dir $SAVE_DIR --max-epoch 50 --task librispeech_task --arch fairseq_las_1 --optimizer adam --lr 1e-04 --max-tokens 20000 --log-format json --log-interval 10 --criterion label_smoothed_cross_entropy_with_acc --user-dir fairseq_las/ --num-workers 4 --tensorboard-logdir $TENSORBOAD_DIR --lr-scheduler reduce_lr_on_plateau --clip-norm 10.0 --save-dir $SAVE_DIR --lr-patience 1 --lr-shrink 0.333
```
  
## Inference for librispeech
```
python fairseq_las/infer.py $DIR_FOR_PREPROCESSED_DATA --task librispeech_task --max-tokens 25000 --nbest 1 --path $MODEL_PATH --beam 20 --results-path $RESULT_PATH --batch-size 40 --gen-subset $SUBSET --user-dir fairseq_las/
```
`Sum/Avg` row from first table of the report has WER  
  
## Requirements and Installation
  
* **For decoding** Install wav2letter component  
  
In decoding, We use [wav2letter](https://github.com/facebookresearch/wav2letter) toolkit.  
To quickly summarize the instructions: first, install [CUDA](https://developer.nvidia.com/cuda-downloads). Then follow these steps:  
```
# additional prerequisites - use equivalents for your distro
sudo apt-get install build-essential cmake libatlas-base-dev libfftw3-dev liblzma-dev libbz2-dev libzstd-dev
# install KenLM from source
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j16
cd ..
export KENLM_ROOT_DIR=$(pwd)
cd ..
# install wav2letter python bindings
git clone https://github.com/facebookresearch/wav2letter.git
cd wav2letter/bindings/python
# make sure your python environment is active at this point
pip install torch packaging
pip install -e .
```  
  
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:  
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
  
## Results  
  
### Training Loss Curve  
<img src="https://github.com/sooftware/Fairseq-Listen-Attend-Spell/blob/main/images/loss.PNG" height=270>  
  
### Learning rate  
<img src="https://github.com/sooftware/Fairseq-Listen-Attend-Spell/blob/main/images/lr.PNG" height=270>  
  
|Model|LM|test-clean|test-other|    
|--|:--:|:--:|:--:|      
|Listen, Attend and Spell|-|5.4|14.5|  
|VGG-Transformer|-|5.8|14.1|      
  
※ Comparison criterion is WER (Word Error Rate)  
※ The result of VGC-Transformer was obtained through Speech Recognition Example of fairseq
