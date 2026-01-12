---
title: "MTGT：多模态医学图像分割"
excerpt: "融合医学影像与文本信息的多模态分割方法"
collection: portfolio
date: 2026-01-12
---

# MTGT：多模态医学图像分割（Multimodal Text-Guided Transformer）

本项目提出了一种 **融合医学图像与文本信息的多模态分割框架（MTGT）**，通过引入文本先验与视觉特征的深度交互，提高模型在复杂医学场景下的分割精度与鲁棒性。  
该方法适用于存在语义描述或报告信息的医学影像分割任务。

---

## 📌 方法概述

- **输入模态**：医学影像（MRI / CT / RGB）+ 文本描述
- **核心思想**：  
  - 利用文本引导视觉特征建模  
  - 在多层级上实现图文信息融合  
  - 提升对边界模糊、小目标与语义不确定区域的建模能力

---

## 🧠 模型框架

<p align="center">
  <img src="https://raw.githubusercontent.com/zlxokok/mtgt.github.io/main/1.png" width="700">
</p>

> **图 1**：MTGT 多模态医学图像分割整体框架示意图

---

## ⚙️ 核心实现

### 1. 分割结果指标计算

在验证阶段，模型输出的预测结果用于计算多种分割评价指标，并将预测掩膜保存为二值图像进行可视化分析。

```python
import torch
import numpy as np
import SimpleITK as sitk
import cv2
import os
import sys

for batch_idx, (sampled_batch, name) in enumerate(valloader):
    images = sampled_batch['image'].cuda().float()
    masks = sampled_batch['label'].cuda()
    text = sampled_batch['text'].cuda()

    masks_pred = model(images, text)
    predicted = masks_pred.argmax(1)

    # 计算评价指标
    mdice = Miou.calculate_mdice(predicted, masks, 2).item()
    miou = Miou.calculate_miou(predicted, masks, 2).item()
    dice = Miou.dice(predicted, masks).item()
    precision = Miou.precision(predicted, masks).item()
    recall = Miou.recall(predicted, masks).item()
    f1 = Miou.F1score(predicted, masks).item()

    # 保存预测结果
    predict = predicted.squeeze(0).cpu().numpy()
    mask_np = (predict * 255).astype('uint8')
    mask_np[mask_np > 0] = 255
    cv2.imwrite(os.path.join(result_path, name[0]), mask_np)
```
### 2. 分割结果可视化

<p align="center">
  <img src="https://raw.githubusercontent.com/zlxokok/mtgt.github.io/main/bjorke_9_img.png" width="45%">
  <img src="https://raw.githubusercontent.com/zlxokok/mtgt.github.io/main/bjorke_9.png" width="45%">
</p>






    
