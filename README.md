#**Automated Image Captioning Using CNN and NLP**

#**Overview**

This repository contains the implementation, experiments, and analysis for an MSc Data Science final project on automated image captioning. The project develops an end-to-end model combining Convolutional Neural Networks (CNNs) for visual feature extraction and Natural Language Processing (NLP) techniques for generating descriptive captions. Using the MS COCO dataset, the model employs a CNN encoder (e.g., ResNet-50) to encode images and an LSTM-based decoder to produce natural language captions.

**Key goals:**

Extract meaningful visual features from images using pre-trained CNNs.
Generate coherent, contextually relevant captions using sequence-to-sequence learning.
Evaluate performance with metrics like BLEU, METEOR, and ROUGE scores.
Explore applications in accessibility, content moderation, and AI-assisted storytelling.

The project achieves competitive BLEU-4 scores (~0.28 on validation), demonstrating effective image-to-text translation while addressing challenges like vocabulary size and long-range dependencies.

#**Key Findings**

Model Performance: Encoder-Decoder achieves BLEU-4=0.28 (val), outperforming baseline Naive CNN+RNN (0.22) by 27%.
Ablations: Adding attention boosts BLEU by 8%; Transformer decoder variant reaches 0.31.
Qualitative Insights: Strong on common scenes (e.g., "outdoor activities"); struggles with rare objects (e.g., "vintage bicycle").
Error Analysis: 15% captions miss spatial relations; future: Incorporate object detection (e.g., Faster R-CNN).

#**Evaluation Metrics Table**

Model Variant,BLEU-1,BLEU-2,BLEU-3,BLEU-4,METEOR,ROUGE-L
Baseline (CNN+LSTM),0.65,0.48,0.35,0.22,0.21,0.38
+Attention,0.70,0.53,0.40,0.27,0.24,0.42
Transformer Decoder,0.72,0.55,0.42,0.31,0.26,0.45

#**Data Sources**

Primary: MS COCO 2014 Dataset – 91 categories, 5 captions/image.
Train: 82,783 images; Val: 40,504 images.

Embeddings: GloVe 6B.200d for word representations.
No human participants; all synthetic evaluations.

#**Methods**

Encoder: Pre-trained ResNet-50 (ImageNet) for 2048-dim features.
Decoder: LSTM with embedding layer; beam search (width=3) for inference.
Training: Adam optimizer, categorical cross-entropy; data aug. (random crops/flips).
Evaluation: NLTK toolkit for BLEU/METEOR; custom ROUGE implementation.
Ethics: Model trained on public data; biases in captions (e.g., gender stereotypes) noted and mitigated via diverse sampling.

#**Limitations and Future Work**

Dataset bias: COCO over-represents Western scenes; extend to multilingual/diverse datasets (e.g., Visual Genome).
Scalability: Single-GPU limits; distribute via TensorFlow strategies.
Enhancements: Integrate CLIP for zero-shot; fine-tune on domain-specific data (e.g., medical images).
