# MachineComprehension
### Introduction
Pytorch implementation of machine comprehension papers for SQuAD v1.1 <br><br>
BIDAF (Minjoon Seo et al.,2016)  https://arxiv.org/abs/1611.01603 <br>
QANet (Adams Wei Yu et al.,2018) https://arxiv.org/abs/1804.09541 <br>
Ensemble model for BIDAF and QANet
### Structure
preproc.py: dataset preprocessing and build training features <br>
config.py: determine which model to train and hype-parameters setting <br>
evaluate.py: evaluate script <br>
main.py: program entry <br>
models/qanet.py: QANet model <br>
models/bidaf.py: BIDAF model <br>
models/ensemble.py QANet and BIDAF ensemble <br>

### Result
Result on dev dataset <br>

|       |  QANet | BIDAF | Ensemble |
| :---: |  :---: | :---: | :------: |
| F1 |  76.3  | 74.1  |  77.6  |
| EM |  67.5  | 63.3 | 68 |
### Difference with papers
* Context length is set to 300 due to limit of memory
* Char embedding doesn't be connected with convolution layer 
* Difference hype parameters setting
### Todo

- [ ] Achieve the result in paper
- [ ]  Reduce memory cost
- [ ]  Complete R-net model
- [ ]  Do the ensemble
