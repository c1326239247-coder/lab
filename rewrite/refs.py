# -*- coding: utf-8 -*-
"""Reference list of the manuscript, keyed by citation label.

Entries are keyed rather than numbered: the body text in content.py cites
@@key@@ markers, and build_docx.py assigns numbers in order of first
appearance and drops nothing silently.  A key that is never cited, or a
citation without an entry here, aborts the build.

Every entry is given with the venue and the persistent identifier
(arXiv id or DOI) of the version we actually consulted.  Volume and page
numbers are quoted only where we could verify them; an unverifiable page
range is omitted rather than guessed.
"""

REFS = [
    ("radford2021clip",
     "Radford A, Kim J W, Hallacy C, et al. Learning transferable visual "
     "models from natural language supervision. In: Proceedings of the 38th "
     "International Conference on Machine Learning (ICML), 2021. "
     "arXiv:2103.00020."),
    ("dosovitskiy2021vit",
     "Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 "
     "words: Transformers for image recognition at scale. In: International "
     "Conference on Learning Representations (ICLR), 2021. arXiv:2010.11929."),
    ("vaswani2017attention",
     "Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. In: "
     "Advances in Neural Information Processing Systems (NeurIPS), 2017, 30. "
     "arXiv:1706.03762."),
    ("touvron2023llama2",
     "Touvron H, Martin L, Stone K, et al. Llama 2: Open foundation and "
     "fine-tuned chat models. arXiv preprint, 2023. arXiv:2307.09288."),
    ("zhang2024tinyllama",
     "Zhang P, Zeng G, Wang T, Lu W. TinyLlama: An open-source small "
     "language model. arXiv preprint, 2024. arXiv:2401.02385."),
    ("su2024roformer",
     "Su J, Ahmed M, Lu Y, et al. RoFormer: Enhanced transformer with rotary "
     "position embedding. Neurocomputing, 2024, 568: 127063."),
    ("ainslie2023gqa",
     "Ainslie J, Lee-Thorp J, de Jong M, et al. GQA: Training generalized "
     "multi-query transformer models from multi-head checkpoints. In: "
     "Proceedings of EMNLP, 2023. arXiv:2305.13245."),
    ("hu2022lora",
     "Hu E J, Shen Y, Wallis P, et al. LoRA: Low-rank adaptation of large "
     "language models. In: International Conference on Learning "
     "Representations (ICLR), 2022. arXiv:2106.09685."),
    ("li2021prefix",
     "Li X L, Liang P. Prefix-tuning: Optimizing continuous prompts for "
     "generation. In: Proceedings of ACL-IJCNLP, 2021. arXiv:2101.00190."),
    ("liu2022ptuningv2",
     "Liu X, Ji K, Fu Y, et al. P-Tuning v2: Prompt tuning can be comparable "
     "to fine-tuning universally across scales and tasks. In: Proceedings of "
     "ACL, 2022. arXiv:2110.07602."),
    ("zhang2024llamaadapter",
     "Zhang R, Han J, Liu C, et al. LLaMA-Adapter: Efficient fine-tuning of "
     "language models with zero-init attention. In: International Conference "
     "on Learning Representations (ICLR), 2024. arXiv:2303.16199."),
    ("jia2022vpt",
     "Jia M, Tang L, Chen B C, et al. Visual prompt tuning. In: Proceedings "
     "of the European Conference on Computer Vision (ECCV), 2022. "
     "arXiv:2203.12119."),
    ("zhou2022coop",
     "Zhou K, Yang J, Loy C C, Liu Z. Learning to prompt for vision-language "
     "models. International Journal of Computer Vision, 2022, 130(9): "
     "2337-2348."),
    ("liu2021swin",
     "Liu Z, Lin Y, Cao Y, et al. Swin Transformer: Hierarchical vision "
     "transformer using shifted windows. In: Proceedings of the IEEE/CVF "
     "International Conference on Computer Vision (ICCV), 2021. "
     "arXiv:2103.14030."),
    ("dong2022cswin",
     "Dong X, Bao J, Chen D, et al. CSWin Transformer: A general vision "
     "transformer backbone with cross-shaped windows. In: Proceedings of the "
     "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), "
     "2022. arXiv:2107.00652."),
    ("xiao2024streamingllm",
     "Xiao G, Tian Y, Chen B, Han S, Lewis M. Efficient streaming language "
     "models with attention sinks. In: International Conference on Learning "
     "Representations (ICLR), 2024. arXiv:2309.17453."),
    ("yao2025cacheblend",
     "Yao J, Li H, Liu Y, et al. CacheBlend: Fast large language model "
     "serving for RAG with cached knowledge fusion. In: Proceedings of the "
     "20th European Conference on Computer Systems (EuroSys), 2025. "
     "doi:10.1145/3689031.3696098."),
    ("li2024kvsurvey",
     "Li H, Li Y, Tian A, et al. A survey on large language model "
     "acceleration based on KV cache management. arXiv preprint, 2024. "
     "arXiv:2412.19442."),
    ("zhou2024dynamickv",
     "Zhou X, Wang W, Zeng M, et al. DynamicKV: Task-aware adaptive KV cache "
     "compression for long context LLMs. In: Findings of the Association for "
     "Computational Linguistics: EMNLP 2025. "
     "doi:10.18653/v1/2025.findings-emnlp.426."),
    ("beltagy2020longformer",
     "Beltagy I, Peters M E, Cohan A. Longformer: The long-document "
     "transformer. arXiv preprint, 2020. arXiv:2004.05150."),
    ("alayrac2022flamingo",
     "Alayrac J B, Donahue J, Luc P, et al. Flamingo: A visual language "
     "model for few-shot learning. In: Advances in Neural Information "
     "Processing Systems (NeurIPS), 2022, 35: 23716-23736. arXiv:2204.14198."),
    ("tsimpoukelli2021frozen",
     "Tsimpoukelli M, Menick J, Cabi S, et al. Multimodal few-shot learning "
     "with frozen language models. In: Advances in Neural Information "
     "Processing Systems (NeurIPS), 2021. arXiv:2106.13884."),
    ("li2023blip2",
     "Li J, Li D, Savarese S, Hoi S. BLIP-2: Bootstrapping language-image "
     "pre-training with frozen image encoders and large language models. In: "
     "Proceedings of the 40th International Conference on Machine Learning "
     "(ICML), 2023. arXiv:2301.12597."),
    ("liu2023llava",
     "Liu H, Li C, Wu Q, Lee Y J. Visual instruction tuning. In: Advances in "
     "Neural Information Processing Systems (NeurIPS), 2023, 36. "
     "arXiv:2304.08485."),
    ("liu2024llava15",
     "Liu H, Li C, Li Y, Lee Y J. Improved baselines with visual instruction "
     "tuning. In: Proceedings of the IEEE/CVF Conference on Computer Vision "
     "and Pattern Recognition (CVPR), 2024. arXiv:2310.03744."),
    ("liu2024llavanext",
     "Liu H, Li C, Li Y, et al. LLaVA-NeXT: Improved reasoning, OCR, and "
     "world knowledge. Technical report, January 2024. "
     "https://llava-vl.github.io/blog/2024-01-30-llava-next/."),
    ("zhu2024minigpt4",
     "Zhu D, Chen J, Shen X, Li X, Elhoseiny M. MiniGPT-4: Enhancing "
     "vision-language understanding with advanced large language models. In: "
     "International Conference on Learning Representations (ICLR), 2024. "
     "arXiv:2304.10592."),
    ("dai2023instructblip",
     "Dai W, Li J, Li D, et al. InstructBLIP: Towards general-purpose "
     "vision-language models with instruction tuning. In: Advances in Neural "
     "Information Processing Systems (NeurIPS), 2023, 36. arXiv:2305.06500."),
    ("bai2023qwenvl",
     "Bai J, Bai S, Yang S, et al. Qwen-VL: A versatile vision-language "
     "model for understanding, localization, text reading, and beyond. arXiv "
     "preprint, 2023. arXiv:2308.12966."),
    ("yin2023mmllmsurvey",
     "Yin S, Fu C, Zhao S, et al. A survey on multimodal large language "
     "models. arXiv preprint, 2023. arXiv:2306.13549."),
    ("goyal2017vqav2",
     "Goyal Y, Khot T, Summers-Stay D, Batra D, Parikh D. Making the V in "
     "VQA matter: Elevating the role of image understanding in visual "
     "question answering. In: Proceedings of the IEEE Conference on Computer "
     "Vision and Pattern Recognition (CVPR), 2017: 6904-6913."),
    ("marino2019okvqa",
     "Marino K, Rastegari M, Farhadi A, Mottaghi R. OK-VQA: A visual "
     "question answering benchmark requiring external knowledge. In: "
     "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern "
     "Recognition (CVPR), 2019: 3195-3204."),
    ("hudson2019gqa",
     "Hudson D A, Manning C D. GQA: A new dataset for real-world visual "
     "reasoning and compositional question answering. In: Proceedings of the "
     "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), "
     "2019: 6700-6709."),
    ("loshchilov2019adamw",
     "Loshchilov I, Hutter F. Decoupled weight decay regularization. In: "
     "International Conference on Learning Representations (ICLR), 2019. "
     "arXiv:1711.05101."),
    ("smith2019superconvergence",
     "Smith L N, Topin N. Super-convergence: Very fast training of neural "
     "networks using large learning rates. In: Artificial Intelligence and "
     "Machine Learning for Multi-Domain Operations Applications, SPIE, 2019. "
     "arXiv:1708.07120."),
    ("efron1993bootstrap",
     "Efron B, Tibshirani R J. An Introduction to the Bootstrap. New York: "
     "Chapman and Hall, 1993."),
    ("ba2016layernorm",
     "Ba J L, Kiros J R, Hinton G E. Layer normalization. arXiv preprint, "
     "2016. arXiv:1607.06450."),
]

BY_KEY = dict(REFS)
ORDER = [key for key, _ in REFS]
