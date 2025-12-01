**<p> 主要功能：<br>**

🖼️ 以图搜图：通过图像相似度检索相似图片<br>
📝 以文搜图：通过文本描述检索相关图片<br>
💬 视觉问答（VQA）：根据图片和问题生成自然语言答案</p><br><br>



**<p> 数据集COCO Dataset (2017)：<br>**
2017 Train images[118K/18G]  #共118287张图片<br>
2017 Val images [5K/1GB]    #共5,000张图片<br>
2017 Train/Val annotations [241MB]   #只用到captions*<br>
captions_train2017.json<br>
captions_val2017.json<br>
</p><br><br>



**<p> 技术实现：<br>**
1. 图文检索<br>
模型：CLIP 的 ViT-B-32 模型提取图像和文本特征向量<br>
检索： Faiss 检索<br>
2. 视觉问答（VQA）<br>
模型：BLIP-2（Salesforce/blip2-opt-2.7b）处理图片和问题，生成自然语言答案
</p><br><br>

