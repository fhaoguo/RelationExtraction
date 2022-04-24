### Steps to transform DuIE to DuNRE-related datasets (these pre-processing steps are independent of other codes of OpenNRE_for_Chinese):
  * Create a folder named "data".
  * Create a folder under "data" named "DuIE".
  * Download the 3 datasets train_data.json, dev_data.json, and all_50_schemas from [BaiDu2019 Relation Extraction Competition](http://lic2019.ccf.org.cn/kg), and put them under a folder named "DuIE".
  * Download the [Tencent AI Lab Embedding Corpus for Chinese Words and Phraseshttps](https://ai.tencent.com/ailab/nlp/embedding.html), and put it under the folder "DuIE".
  * Create a folder under "data" named "DuNRE" to store re-formatted datasets.
  * Run gen_DuNRE.py to create 3 new datasets train_nre.json, dev_nre.json, and relation_nre.json.
  * Run gen_embed_mat.py to create the embedding dictionary file word_dictionary_nre.json.
