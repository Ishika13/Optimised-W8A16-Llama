# Llama-3.2 Model Quantization with W8A16 Linear Layers

This project enhances the efficiency of the `Llama-3.2-1B-Instruct` model by replacing standard linear layers with a custom quantized layer (`W8A16LinearLayer`). The quantized layer stores weights in 8-bit integers (W8) while using 16-bit activations (A16), reducing memory usage and improving inference speed.

## Overview

- **Model Loading:** Loads a pre-trained Llama model with Hugging Face’s `transformers` library.
- **Custom Linear Layer:** Defines `W8A16LinearLayer` that quantizes weights to 8-bit and rescales them during inference.
- **Layer Replacement:** Replaces all `nn.Linear` layers with quantized versions, excluding `lm_head` for prediction accuracy.
- **Text Generation:** Performs text generation before and after quantization for comparison.

## Key Features

- Faster inference with quantized weights.
- Reduced memory footprint using 8-bit integer weights.
- Minimal accuracy degradation with proper scaling.
