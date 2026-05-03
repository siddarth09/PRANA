# Assessment #2: ML-based Vision
## OpenVLA: An Open-Source Vision-Language-Action Model

**Name:** Siddarth

**Course:** CS5330: Pattern Recognition and Computer vision

**Video Link:** [OPENVLA](https://drive.google.com/file/d/1c5rORu2nNMZDksUnSjHlhZM9GNEIXmid/view?usp=sharing)

---

## Reflection

I chose to present on OpenVLA because it sits at an exciting intersection of topics we covered in class (transformers, attention, transfer learning, image classification vs. detection) and a rapidly growing area of robotics research: Vision-Language-Action models. As someone studying robotics, I found it compelling how the same patch-as-token approach from Vision Transformers, the same self-attention mechanism we studied, can be extended beyond image classification to directly control a physical robot arm.

The most challenging part of preparing this presentation was distilling the architecture into something that would make sense to an audience familiar with CNNs and transformers but not necessarily with robot learning. I spent time understanding why the authors chose a dual vision encoder (SigLIP for semantic features, DINOv2 for spatial features) and how the action tokenization scheme elegantly reuses the LLM's existing next-token prediction framework. I think the key insight, that robot actions can be treated as just another token modality alongside vision and language, is what makes VLAs such a natural extension of the transformer paradigm we've been studying.

One thing that surprised me was how much the training procedure differed from typical LLM training: 27 epochs versus the usual one or two, and the finding that 4-bit quantization actually outperforms 8-bit due to inference speed rather than accuracy. These details made me appreciate how much domain-specific engineering goes into adapting foundation models for robotics.

---

## Acknowledgements

### Resources Used
- Kim, M.J., Pertsch, K., Karamcheti, S., et al. "OpenVLA: An Open-Source Vision-Language-Action Model." CoRL 2025. [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- OpenVLA project page: [https://openvla.github.io](https://openvla.github.io)
- OpenVLA GitHub repository: [https://github.com/openvla/openvla](https://github.com/openvla/openvla)
- Wikipedia article on Vision-Language-Action models for broader context on the VLA landscape

### LLM Usage
I used Claude (Anthropic) as a learning aid during the preparation of this presentation. Specifically, Claude helped me:
- **Understand concepts from the paper:** I discussed the OpenVLA architecture, the rationale behind the dual vision encoder design, the action tokenization scheme, and the training procedure with Claude to deepen my understanding before creating the presentation.
- **Slide editing:** Claude assisted visual design of the slides.
- **Organizing talking points:** After I understood the material, Claude helped me structure my speaking notes to fit within the 10-minute time limit.




