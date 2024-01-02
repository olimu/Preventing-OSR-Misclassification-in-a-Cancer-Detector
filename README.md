# Preventing-OSR-Misclassification-in-a-Cancer-Detector
**Mitigating Open-Set Misclassification in a Colorectal Cancer Detecting Neural Network**

Though AI models successfully classify objects they were trained on, they struggle when they encounter unfamiliar outlier objects that they weren’t trained on, which are prevalent in real world applications. Consider an algorithm trained to classify tumors as cancerous or non-cancerous. This algorithm successfully differentiates between tumors, however, when it comes across an unfamiliar non-tumorous tissue, like cancerous stroma, it misdiagnoses the patient. It’s impossible to train algorithms to recognize every tissue, so I developed Intra-Dataset Outlier Exposure (IDOE).
 
IDOE exposes algorithms to outlier examples during training so that they learn to recognize intrinsic outlier characteristics. It then generalizes these traits to detect new, unfamiliar outliers. By showing the tumor-classifying algorithm examples of non-tumorous outliers, like mucus and stroma, the algorithm learns to classify all non-tumorous tissues as outliers, like cancerous stroma.

I found that IDOE achieves an average score of 94.8% in preventing the misdiagnosis of healthy and cancerous colon tissues or tumors, so it likely will be effective in similar medical diagnosis applications.

**code folder**
* ood: MNIST  
  * ood_thres_nrand.py generates the randomized NN models  
  * ood_control_nrand.py generates the randomized control NN models  
  * ood_utils.py contains the support routines  
* med: medMNIST  
  * med_thres_nrand.py generates the randomized NN models  
  * med_control_nrand.py generates the randomized control NN models  
  * med_utils.py contains the support routines  
  * dataset_without_pytorch.py contains medMNIST data loading support routines  
* ncheck_thres.py used for analysis of the effect of different thresholds  
uses thres.py for support routines  
or argmaxisthres.py for support routines

**Preventing-OSR-Misclassification-Paper.pdf** paper submitted to the National Science Talent Search Competition

**Preventing-OSR-Misclassification-Poster.pdf** poster from the Pittsburgh Regional Science and Engineering Fair

**Preventing-OSR-Misclassification-Slides.pdf:** slides from the Pennsylvania Junior Academy of Science competition where I won 1st place at the Region 7 and State Pennsylvania Junior Science Academy competitions in the Computer Science category


