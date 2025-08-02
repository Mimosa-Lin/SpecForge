import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)


def initialize_past_key_values(model, max_length=2200):
    """
    Initialize past key and value states for LlavaForConditionalGeneration (LLaMA-based).
    
    Args:
        model (LlavaForConditionalGeneration): The transformer model.
        max_length (int): Maximum sequence length for KV cache.

    Returns:
        tuple: (past_key_values, past_key_values_data_list, current_length_data)
    """
    config = model.config
    batch_size = 1

    llama_layers = model.model.language_model.layers
    num_layers = len(llama_layers)
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    devices = [layer.self_attn.q_proj.weight.device for layer in llama_layers]

    past_key_values_data_list = []
    start_device = devices[0]
    start_index = 0

    for idx, device in enumerate(devices + [None]): 
        if idx == len(devices) or devices[idx] != start_device:
            num_layers_on_device = idx - start_index
            kv_data = torch.zeros(
                num_layers_on_device * 2,
                batch_size,
                num_key_value_heads,
                max_length,
                head_dim,
                device=start_device,
                dtype=model.dtype
            )
            past_key_values_data_list.append(kv_data)
            if idx < len(devices):
                start_device = devices[idx]
                start_index = idx

    current_length_data = torch.zeros(num_layers * 2, dtype=torch.long, device="cpu")
    past_key_values = []
    layer_offset = 0
    current_device = devices[0]
    buffer_index = 0

    for i, device in enumerate(devices):
        if device != current_device:
            buffer_index += 1
            layer_offset = 0
            current_device = device

        kv_cache = [
            KVCache(past_key_values_data_list[buffer_index][2 * layer_offset + j],
                    current_length_data[i * 2 + j])
            for j in range(2)  # key and value
        ]
        past_key_values.append(kv_cache)
        layer_offset += 1

    return past_key_values, past_key_values_data_list, current_length_data