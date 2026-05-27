"""Local vLLM engine wrapper for quantized model inference.

This module provides production-ready vLLM integration for running large language models
locally with minimal latency and memory overhead. It supports quantized models (AWQ, GPTQ)
and uses vLLM's PagedAttention kernel for efficient batch processing of multiple completions.

PagedAttention enables efficient memory management for large batch sizes by allowing the
key-value (KV) cache to be paged, reducing memory fragmentation and enabling higher
throughput. This is particularly useful when generating multiple `n` completions per
prompt, as vLLM handles all `n` sequences in a single inference pass.

Example:
    from src.vllm_engine import LocalLLMEngine
    
    engine = LocalLLMEngine()
    # Generate 15 responses for each of 2 problems
    outputs = engine.generate_batch(
        problem_texts=['Compute 2+2', 'Solve x=5'],
        n=15
    )
    # outputs[0] = [response_1, response_2, ..., response_15] for 'Compute 2+2'
    # outputs[1] = [response_1, response_2, ..., response_15] for 'Solve x=5'
"""

from typing import List


class LocalLLMEngine:
    """Production-ready vLLM engine for local inference.
    
    Features:
    - Quantization support (AWQ, GPTQ) for reduced memory footprint
    - Efficient batch processing with PagedAttention
    - Configurable tensor parallelism for multi-GPU setups
    - GPU memory utilization tuning
    """

    def __init__(
        self,
        model: str = 'Qwen/Qwen2.5-Math-7B-Instruct-AWQ',
        quantization: str = 'awq',
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ):
        """Initialize the vLLM engine with model configuration.
        
        Args:
            model: Model path or HuggingFace model ID (e.g., 'Qwen/Qwen2.5-Math-7B-Instruct-AWQ')
            quantization: Quantization format ('awq', 'gptq', or None for full precision)
            tensor_parallel_size: Number of GPUs for tensor parallelism (default 1 for single GPU)
            gpu_memory_utilization: GPU memory utilization fraction (0.0-1.0). Higher values
                                   allow for larger batch sizes at the cost of potential OOM.
        """
        self.model_name = model
        self.quantization = quantization
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_memory_utilization = float(gpu_memory_utilization)

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vllm is required for LocalLLMEngine. Install with:\n"
                "  pip install vllm\n"
                "See https://github.com/vllm-project/vllm for detailed setup instructions."
            ) from e

        self._SamplingParams = SamplingParams
        self.llm = None

        # Initialize the vLLM model with production-ready settings
        try:
            self.llm = LLM(
                model=self.model_name,
                quantization=self.quantization if self.quantization else None,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                # Additional optimizations
                trust_remote_code=True,
                dtype='float16',  # Use float16 for better throughput
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize vLLM model '{self.model_name}': {e}\n"
                f"Ensure the model is available on HuggingFace Hub and your GPU has "
                f"sufficient memory for quantization={self.quantization}."
            ) from e

    def generate_batch(self, problem_texts: List[str], n: int = 15) -> List[List[str]]:
        """Generate multiple responses for each problem using vLLM batch inference.
        
        Uses vLLM's SamplingParams with n>1 to request multiple completions per prompt.
        PagedAttention efficiently handles all n sequences in a single forward pass,
        significantly reducing memory overhead compared to sequential generation.
        
        Args:
            problem_texts: List of problem prompts (e.g., math questions in plain text)
            n: Number of completions to generate per problem (default 15 for self-consistency)
        
        Returns:
            List[List[str]]: For each input problem, returns a list of n generated text responses.
                           The outer list has len(problem_texts) elements; each inner list has n strings.
        
        Example:
            >>> texts = ['Compute 2+2', 'Solve x+1=5']
            >>> outputs = engine.generate_batch(texts, n=3)
            >>> len(outputs)  # 2
            >>> len(outputs[0])  # 3 (three completions for first problem)
        """
        if self.llm is None:
            raise RuntimeError("vLLM model is not initialized")

        # Set up sampling parameters for diverse completions
        # temperature=0.7 encourages diversity; higher n improves self-consistency
        try:
            sampling_params = self._SamplingParams(
                temperature=0.7,
                top_p=0.95,
                n=n,
                max_tokens=1024,
                use_beam_search=False,  # Faster than beam search; suitable for n>1
            )
        except TypeError as e:
            raise RuntimeError(
                f"Failed to construct SamplingParams with n={n}: {e}\n"
                "This may indicate an incompatibility with your vLLM version."
            ) from e

        try:
            # vLLM handles batching internally; pass all prompts at once for efficiency
            # PagedAttention manages KV cache for all n sequences per prompt
            outputs = self.llm.generate(problem_texts, sampling_params)
        except Exception as e:
            raise RuntimeError(
                f"vLLM generation failed for {len(problem_texts)} prompts with n={n}: {e}"
            ) from e

        # Extract and organize completions: outputs[i].outputs contains all n completions
        # for the i-th prompt
        results: List[List[str]] = []
        try:
            for output in outputs:
                completions = []
                # Each output.outputs[j] is a CompletionSequence with .text attribute
                if hasattr(output, 'outputs') and isinstance(output.outputs, list):
                    for completion in output.outputs:
                        text = getattr(completion, 'text', str(completion))
                        completions.append(text)
                else:
                    # Fallback: treat output as single completion and repeat
                    text = getattr(output, 'text', str(output))
                    completions = [text] * n
                
                results.append(completions)
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract generated text from vLLM outputs: {e}\n"
                "Output structure may have changed in your vLLM version."
            ) from e

        return results
