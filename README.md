
#### Description of different files

* src folder: contains modified source code from LXMERT to generate the input features.
* transformer.py: code for the generator class and the training and evaluation functions. 
* modules.py: modules definition.
* executor.py: the executor class with its training and evaluation functions.
* tree.py: validity constrains definition and validity matrix instantiation.

#### Feature extraction
* We use LXMERT to extract image and question features, for each question/image pair we forward it through LXMERT to get the joint features, the features are saved as tensors s.t q_id.pth is the file represnting the features for the question q_id.
 the tensors have size (65, 768), 768 is the number of feature dimension, the first 29 lines represent the question features (29 words) and the last 36 lines represnt the 36 objects bboxes features.

  * download the dataset .tsv files of the redistribution of the gqa dataset from LXMERT github, for more details check LXMERT documentation.
    wget https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip -P data/vg_gqa_imgfeat
    unzip data/vg_gqa_imgfeat/vg_gqa_obj36.zip -d data && rm data/vg_gqa_imgfeat/vg_gqa_obj36.zip
    wget https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/gqa_testdev_obj36.zip -P data/vg_gqa_imgfeat
    unzip data/vg_gqa_imgfeat/gqa_testdev_obj36.zip -d data && rm data/vg_gqa_imgfeat/gqa_testdev_obj36.zip
  * cd src then call python3 main_feats.py --train=train
  * The *src* files content is from https://github.com/airsplay/lxmert wth some modifications.

#### Generator training:

  * mkdir save, the folder in which you want to save the outputs loss, models, figures ...
  * training and evaluation are coded in the *transformer.py* file, when training and evaluating the model we use teacher forcing to make the training easier. As for the *Predictor* class in the *predictor.py* file we redefine the prediction functions without teacher forcing. 
Inputs: the generator gets question/layout pairs as training data. 
  * First, in the *transformer.py* file we redefine the *TransformerDecoderLayer* class (https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html) in order to get the *attn_output_weights *from the multihead attention next the output. 
  * The *attn_output_weights* are used for the calculation of the textual parameter when using the Predictor class.
  * Next, we define the *TransformerDataset* class to read the inputs and return the data in the appropriate format for batch learning. This is a redefinition of the Dataset class from pytorch.
  * Then, the ProgramGenerator class redefines the forward method from *nn.Module* and calls the TransformerDecoderLayer class.
  * The *train_epoch* reads the data batches and executes one epoch over all the data batches, the loss is computed for this learning step and backwarded to optimize the model.
  * The *train* function loops over the train_epoch function in order to train the model for several epochs, save the checkpoints and plots.
  * The *evaluate* function runs a forward pass on the model without backprobagation, useful for printing the outputs of some examples or evaluating the outcome of a specific epoch.
  * The *main* is at the end of the file, we use it to call the train or the evaluate methods, examples are provided in the file.
  * To run the *transformer.py* from scratch call: python3 transformer.py --train=train --data_path=_ --transformer_save=_ --transformer_lr=0.1 --transformer_start=0  --transformer_path=""
  * To resume training from a previous epoch call: python3 transformer.py --train=train --data_path=_ --transformer_save=_ --transformer_lr=0.1 --transformer_start=100  --transformer_path="--data_path=_/transformer_100.pth"
  * --train: the dataset to use: train, validation or testdev.
  
#### Executor training:
  
  Inputs: the executor takes layout/answer pairs.
  * We create the *ExecutorDataset* class to load the data.
  * The *Executor* class extends the *nn.Module* class and redefines the *forward* method in different manners, when intializing the Executor instance it creates the different modules at the same time in the init method and puts them in the *modules* dictionary for easy access.
  * To run the training of the executer: python3 executor.py --executor_lr=0.1 --executor_save=_ --executor_start=0  --split=train --executor_bs=1024 --seed=2 --functions=coordinates --use_coordinates=True  --weighting=0.1,0.7,0.2 --loss_reduction=none --tgt_att_type=soft


## References


```bibtex
@conference{visapp23,
author={Wafa Aissa. and Marin Ferecatu. and Michel Crucianu.},
title={Curriculum Learning for Compositional Visual Reasoning},
booktitle={Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2023) - Volume 5: VISAPP},
year={2023},
pages={888-897},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0011895400003417},
isbn={978-989-758-634-7},
issn={2184-4321},
}
```

```bibtex
@InProceedings{10.1007/978-3-031-45382-3_30,
author="Aissa, Wafa and Ferecatu, Marin and Crucianu, Michel",
title="Multimodal Representations for Teacher-Guided Compositional Visual Reasoning",
booktitle="Advanced Concepts for Intelligent Vision Systems",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="357--369",
isbn="978-3-031-45382-3"
}
```
  
