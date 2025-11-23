多模态检索与问答系统，主要功能包括：以图搜图、以文搜图、视觉问答。

###################################################

数据集：https://cocodataset.org/#download

2017 Train images[118K/18G]  #共118287张图片

2017 Val images [5K/1GB]    #共5,000张图片

2017 Train/Val annotations [241MB]   #只用到captions*

captions_train2017.json

captions_val2017.json

###################################################


图文检索：CLIP 的ViT-B-32做向量归一化，Faiss进行检索。

VQA：BLIP-2 模型（Salesforce/blip2-opt-2.7b）处理图片和问题，生成自然语言答案。
