# -*- coding: utf-8 -*-
# 请先安装依赖包：open-clip-torch pillow faiss-cpu torch numpy
# 可选：transformers(用于VQA)
# 以图搜图用法：python search_demo.py 000000003192.jpg 5


import os, sys, json
import torch
import numpy as np
from PIL import Image
import faiss
import open_clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_BASE_URL", "https://hf-mirror.com")

# os.environ["HF_HUB_OFFLINE"] = "1"

# 数据路径（image和caption）
IMG_DIR  = "/Users/xxx/Downloads/train2017"
CAP_JSON = "/Users/xxx/Downloads/annotations/captions_preview.json"

# 读取JSON，聚合相同image_id的captions
def load_pairs(caption_json_path, img_dir):
    with open(caption_json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data:
        anns = data["annotations"]
    elif isinstance(data, list):
        anns = data
    else:
        raise ValueError("JSON 格式不对")

    buckets = {}
    for a in anns:
        iid = a["image_id"]
        cap = a.get("caption", "").strip()
        if not cap:
            continue
        buckets.setdefault(iid, []).append(cap)

    pairs = []
    for iid, caps in buckets.items():
        fpath = os.path.join(img_dir, f"{iid:012d}.jpg")
        if os.path.exists(fpath):
            pairs.append({"image_id": iid, "path": fpath, "captions": caps})

    if not pairs:
        print("没有找到有效的‘图片-caption’对，请检查路径")
        sys.exit(1)

    return pairs

# 加在OpenCLIP模型与预处理，优先选择GPU
def load_model():
    device = pick_device()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device

# 对图像库中的所有图片提取、归一化
def encode_gallery(pairs, model, preprocess, device):
    img_feats, img_meta = [], []
    with torch.inference_mode():
        for p in pairs:
            img = Image.open(p["path"]).convert("RGB")
            img_t = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(img_t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            img_feats.append(feat.cpu().numpy()[0])
            img_meta.append(p)
    img_feats = np.stack(img_feats).astype("float32")
    return img_feats, img_meta

# 建立FAISS内积索引
def build_index(img_feats):
    d = img_feats.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(img_feats)
    return index

# 对要查询的图片提取归一化向量
def encode_query_image(query_path, model, preprocess, device):
    if not os.path.exists(query_path):
        print(f"错误: 查询图片 {os.path.basename(query_path)} 不存在于 {os.path.dirname(query_path)}")
        sys.exit(1)
    with torch.no_grad():
        qimg = Image.open(query_path).convert("RGB")
        qimg_t = preprocess(qimg).unsqueeze(0).to(device)
        qfeat = model.encode_image(qimg_t)
        qfeat = qfeat / qfeat.norm(dim=-1, keepdim=True)
        qfeat = qfeat.cpu().numpy().astype("float32")
    return qfeat

# （新增）把一段文本编码成查询向量
def encode_query_text(query_text, model, tokenizer, device):
    with torch.no_grad():
        tok = tokenizer([query_text]).to(device)
        tfeat = model.encode_text(tok)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        tfeat = tfeat.cpu().numpy().astype("float32")
    return tfeat

# 检索并且打印结果
def search_and_print(index, qvec, img_meta, top_k, header="检索结果 TopK"):
    if top_k < 1:
        print("TopK 必须 >= 1")
        sys.exit(1)
    if top_k > len(img_meta):
        print(f"警告: TopK={top_k} 超过库大小({len(img_meta)}), 自动设置为 {len(img_meta)}")
        top_k = len(img_meta)

    D, I = index.search(qvec, k=top_k)
    print(f"\n{header}:")
    for rank, idx in enumerate(I[0]):
        meta = img_meta[idx]
        print(f"#{rank+1} | {meta['path']} | caption示例: {meta['captions'][0]} | 相似度={D[0][rank]:.4f}")

# 选设备（CUDA > MPS > CPU）
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def vqa_blip2(image_path, question, model_id="Salesforce/blip2-opt-2.7b", max_new_tokens=30):
    device = pick_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    cache_dir = os.path.expanduser("~/.cache/huggingface")
    offline = os.getenv("HF_HUB_OFFLINE", "0") == "1"

    try:
        # 优先本地缓存
        processor = Blip2Processor.from_pretrained(model_id, local_files_only=True, cache_dir=cache_dir)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=True, cache_dir=cache_dir
        ).to(device)
    except Exception as e:
        if offline:
            raise RuntimeError(
                "[VQA] 处于离线模式，但本地未找到模型缓存。\n"
                "解决：临时关闭离线并运行一次脚本完成预热，或用 snapshot_download 先把模型拉到本地缓存。"
            ) from e
        print("[VQA] 本地缓存未找到，尝试联网下载一次（成功后未来可离线使用）...")
        processor = Blip2Processor.from_pretrained(model_id, cache_dir=cache_dir)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=cache_dir
        ).to(device)

    image = Image.open(image_path).convert("RGB")
    clean_q = question.strip()
    prompt = f"Question: {clean_q}\nAnswer:"

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.35,
        no_repeat_ngram_size=3,
        early_stopping=True,
        length_penalty=0.0,
    )

    # 设置 eos_token_id，保证答案干净收尾
    try:
        eos_id = getattr(processor.tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(getattr(model.config, "text_config", None), "eos_token_id", None)
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
    except Exception:
        pass

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    answer = text.split("Answer:", 1)[-1].strip()
    if answer.lower().startswith("question:"):
        answer = answer.split("answer:", 1)[-1].strip()

    print(f"\n[VQA] Q: {question}\n     A: {answer}")
    return answer


def main():
    if len(sys.argv) < 3:
        print("用法: python search_demo.py <图片文件名> <TopK>")
        print("用法2（以图+以文）: python search_demo.py <图片文件名> <TopK> <文本查询...>") # 新增
        print("用法3（附带VQA）  : python search_demo.py <图片文件名> <TopK> <文本查询...> --vqa <你的问题>")
        sys.exit(1)

    query_fname = sys.argv[1]
    top_k = int(sys.argv[2])

    # 解析第3个参数起：可含文本检索与 --vqa
    tail = sys.argv[3:] if len(sys.argv) > 3 else []
    vqa_question = None
    if "--vqa" in tail:
        vqa_pos = tail.index("--vqa")
        text_query = " ".join(tail[:vqa_pos]).strip() if vqa_pos > 0 else None
        vqa_question = " ".join(tail[vqa_pos+1:]).strip() or None
    else:
        text_query = " ".join(tail).strip() if tail else None


    # 加载数据
    pairs = load_pairs(CAP_JSON, IMG_DIR)
    query_path = os.path.join(IMG_DIR, query_fname)
    print(f"查询图片: {query_path}, TopK={top_k}")
    print(f"[INFO] 样本图像数: {len(pairs)}")

    # 模型与索引
    model, preprocess, tokenizer, device = load_model()
    img_feats, img_meta = encode_gallery(pairs, model, preprocess, device)
    index = build_index(img_feats)

    # 查询与检索
    qvec = encode_query_image(query_path, model, preprocess, device)
    search_and_print(index, qvec, img_meta, top_k)

    # （新增）以文搜图
    if text_query:
        qvec_txt = encode_query_text(text_query, model, tokenizer, device)
        search_and_print(index, qvec_txt, img_meta, top_k, header=f"【以文搜图】TopK（查询文本：{text_query}）")


    # VQA：优先对“以文搜图”的 Top-1 结果做问答；没有文本就用“以图搜图”的 Top-1
    if vqa_question:
        if text_query:
            # 对文本检索的 Top-1 做 VQA
            Dtxt, Itxt = index.search(qvec_txt, k=1) # 使用 CLIP 的文本编码器
            vqa_path = img_meta[Itxt[0][0]]["path"]
        else:
            # 没有文本时，对以图检索的 Top-1
            Dimg, Iimg = index.search(qvec, k=1)
            vqa_path = img_meta[Iimg[0][0]]["path"]

        vqa_blip2(vqa_path, vqa_question) # 使用 BLIP-2 模型（Salesforce/blip2-opt-2.7b）处理图片和问题

if __name__ == "__main__":
    main()









