In acest repozitoriu se afla lucrarea de licenta a studentului Mihulca Tudor-Octavian, intitulata **Fuziune Multimodala Imagine-Text pentru Diagnosticul Radiografiilor Toracice**. 

Mai jos aveti repartizarea fisierelor:

1. **Preprocesare si Configurare Date**
   - `configurare_dataset_final.py`

2. **Antrenarea Modelelor Unimodale**
   - `antrenare_cnn_gradcam.py`
   - `antrenare_nlp_final.py`

4. **Antrenarea Modelelor de Fuziune (Multimodale)**
   - `fusion_vectori.py`
   - `fusion_inghetare.py`

5. **Evaluarea Modelelor**
   - `evaluare_cnn.py`
   - `evaluare_nlp.py`
   - `evaluare_fusion_vector.py`
   - `evaluare_fusion_inghetare.py`

6. **Modele Salvate**
   - `mimic_cnn_model_best_1.pth`
   - `best_cxr_bert_model_3.pth`
   - `best_fusion_mlp_offline.pth`
   - `best_end2end_frozen.pth`
