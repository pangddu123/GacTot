from typing import Any, Dict, List, Set, Tuple
from transformers import AutoTokenizer  # <--- 新增引入
import ray
import yaml
import torch
from .ray_actor import get_remote_model_generator_class
from .gac_gen_utils import *


def setup_model_actors_and_data(config: List[Dict], norm_type: str, threshold: float) -> Tuple[
    List[Any], List[Any], Any, List[torch.Tensor], Dict[int, str], Dict[Any, str], List[Dict[str, int]], int, List[
        str], int, float
]:
    """
    Sets up model actors based on configurations and preprocesses necessary data for ToT (Token Translation).
    """

    # 1. 配置分数归一化 (保留原 GaC 逻辑)
    update_scores(config, norm_type)
    config = normalize_scores(config)
    logger.info(f"Model ensemble weights: {[(c['name'], round(c['score'], 4)) for c in config]}")

    # 2. 确定主模型 (Primary Model)
    # ToT 必须有一个基准空间，即主模型的 Token 空间
    primary_index = check_priorities(config)
    if primary_index == -1:
        # 如果没有指定 primary，默认第一个模型为主模型
        primary_index = 0
        logger.info(
            f"No 'primary' priority found. Defaulting to the first model ({config[0]['name']}) as the Main Model for ToT.")
    else:
        logger.info(f"ToT Main Model identified: {config[primary_index]['name']} (Index {primary_index}).")

    # 阈值逻辑
    real_threshold = threshold * config[primary_index]["score"] if threshold < 1.0 else 1.0

    config = validate_and_update_quantization(config)

    # 3. 初始化 Ray Model Actors
    # 注意：ToT 中只有 Main Model 必须开启 Cache (如果是 Threshold 模式)，
    # 但如果是 Every Step 模式，所有模型都应该开启 Cache 以提高效率。
    # 这里沿用原逻辑：如果有 primary 且 threshold 生效，则其他模型可能不用 cache；否则都用。
    # 为了简化，建议 ToT 模式下所有模型都开启 use_cache=True，除非显存极度紧张。
    model_actors_list = [
        get_remote_model_generator_class(model_config["num_gpus"]).remote(
            model_path=model_config["weight"],
            max_memory=model_config["max_memory"],
            model_name=model_config["name"],
            model_ensemble_weight=model_config["score"],
            use_cache=True,
            quantization=model_config["quantization"],
            # 新增：从配置中获取 enable_thinking，如果没有则默认为 None
            enable_thinking=model_config.get("enable_thinking", None)
        )
        for i, model_config in enumerate(config)
    ]

    # 4. 获取所有 Tokenizers 和 Model Names
    # 4. 获取所有 Tokenizers 和 Model Names
    # --- 修改开始：在本地加载 Tokenizer，避免 Ray 序列化错误 ---
    tokenizers = []
    for model_config in config:
        model_path = model_config["weight"]
        model_name = model_config["name"]

        # 复制 ray_actor.py 中的加载逻辑 (例如 Yi 模型需要 use_fast=False)
        use_fast = True
        if model_name in ["Yi-34B-Chat", "Yi-6B-Chat"]:
            use_fast = False

        logger.info(f"Loading tokenizer locally for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            use_fast=use_fast,
            trust_remote_code=True
        )

        # 简单的修正，确保 pad_token 存在
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizers.append(tokenizer)
    # --- 修改结束 ---

    model_name_list = ray.get([actor.get_model_name.remote() for actor in model_actors_list])

    main_tokenizer = tokenizers[primary_index]

    # 5. 并行构建 ToT 映射矩阵 (使用 Ray Remote)
    # Matrix[i] = Mapping from Model[i] (Assist) to Model[primary] (Main)
    logger.info("Starting ToT Mapping Matrix construction (Parallel)...")

    mapping_matrix_refs = []
    for i, tokenizer in enumerate(tokenizers):
        if i == primary_index:
            # 主模型不需要映射 (Identity)，用 None 占位，推理时跳过计算
            mapping_matrix_refs.append(None)
        else:
            logger.info(f"Dispatching matrix build task: {model_name_list[i]} -> {model_name_list[primary_index]}")
            # 调用 remote function
            # --- 修改开始：传递路径而非对象 ---
            assist_path = config[i]["weight"]
            main_path = config[primary_index]["weight"]

            ref = create_tot_mapping_matrix_remote.remote(
                assist_model_path=assist_path,
                main_model_path=main_path,
                alpha=0.5,
                batch_size=4096
            )
            mapping_matrix_refs.append(ref)

    # 等待所有矩阵构建完成
    # 注意：None 的 ref 不需要 wait
    real_refs = [r for r in mapping_matrix_refs if r is not None]
    if real_refs:
        finished_matrices = ray.get(real_refs)

    # 重组列表顺序
    mapping_matrices = []
    mat_idx = 0
    for ref in mapping_matrix_refs:
        if ref is None:
            mapping_matrices.append(None)
        else:
            # 这里的矩阵是在 CPU 上的，我们在使用前将其移到 GPU (通常在推理函数中做，或者这里做)
            # 建议这里先保留在 CPU，防止显存碎片，推理时按需 .to("cuda")
            mapping_matrices.append(finished_matrices[mat_idx])
            mat_idx += 1

    logger.info("All ToT Mapping Matrices constructed successfully.")

    # 6. 准备辅助数据
    # 特殊前缀 Token (用于反向将 Token ID 转换回 Assist 模型 ID)
    special_prefix_tokens_dict = get_special_prefix_tokens_for_all(tokenizers)

    # Byte Mappings (用于处理二进制 Token)
    byte_mappings_list = [check_byte_mappings(t) for t in tokenizers]

    # 获取 Main Model 的 Vocab Map 用于日志显示 (ToT 最终输出的是 Main Model 的 Token)
    main_vocab = main_tokenizer.get_vocab()
    # 反转字典: ID -> Token String
    index_to_vocab = {v: k for k, v in main_vocab.items()}

    # 最小最大位置编码 (用于截断)
    min_max_position_embeddings = min(
        ray.get(actor.get_max_position_embeddings.remote())
        for actor in model_actors_list
    )

    # 返回值：注意 vocab_union 设为 None，因为 ToT 不需要并集词表
    return (
        model_actors_list,
        tokenizers,
        None,  # vocab_union (Deprecated in ToT)
        mapping_matrices,
        index_to_vocab,  # Main Model's vocab map
        special_prefix_tokens_dict,
        byte_mappings_list,
        min_max_position_embeddings,
        model_name_list,
        primary_index,
        real_threshold,
    )





def validate_and_update_quantization(model_config: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Validates the 'quantization' field in each dictionary of a list of model configurations,
    and adds the 'quantization' field with a default value of 'none' if it's missing.

    Args:
        model_config (List[Dict[str, str]]): 
            A list of dictionaries, where each dictionary represents a model configuration.
            Each dictionary should contain a 'quantization' key, which must have one of the 
            following values: 'none', '8bit', or '4bit'. If the 'quantization' key is missing,
            it will be added with a default value of 'none'.

    Raises:
        ValueError: If any 'quantization' value is not one of 'none', '8bit', or '4bit'.

    Returns:
        List[Dict[str, str]]: The updated list of model configurations with valid 'quantization' values.
    """
    
    # Define the valid quantization options
    valid_quantization_values = {'none', '8bit', '4bit'}
    
    # Loop through each configuration in the input list
    for idx, config in enumerate(model_config):
        # Check if 'quantization' key exists, if not, set it to 'none'
        if 'quantization' not in config:
            config['quantization'] = 'none'
        
        # Get the 'quantization' value
        quantization_value = config['quantization']
        
        # Check if the value is valid, otherwise raise an error with details
        if quantization_value not in valid_quantization_values:
            raise ValueError(
                f"Invalid quantization value '{quantization_value}' in config at index {idx}. "
                f"Allowed values are: {valid_quantization_values}"
            )
    
    # Return the updated list of configurations
    return model_config

def check_priorities(dict_list):
    """
    Check the list of dictionaries to ensure that there is exactly one "primary" priority and all priorities are valid.

    Args:
    dict_list (list of dict): A list where each item is a dictionary with a key "priority" whose value should be either "supportive" or "primary".

    Returns:
    int: Index of the first dictionary with "primary" as priority if there is exactly one, otherwise returns -1.
    """
    allowed_priorities = ["supportive", "primary"]
    primary_index = -1
    primary_count = 0

    for index, d in enumerate(dict_list):
        priority = d.get("priority")

        # Check if the priority is within the allowed values
        if priority not in allowed_priorities:
            raise ValueError(f"'priority' value '{priority}' at index {index} is not allowed!")

        # Check for primary priority and count them
        if priority == "primary":
            primary_count += 1
            if primary_count == 1:
                primary_index = index

    # Warn if there is more than one primary priority
    if primary_count > 1:
        raise ValueError("More than one 'primary' found!")

    return primary_index


def normalize_scores(config, n=1):
    """
    Normalizes the scores of each configuration in the list of dictionaries by multiplying each score by n,
    and then normalizing these scores to a 0 to 1 range such that their sum is 1.
    
    Parameters:
        config (list of dict): A list of dictionaries, each representing a configuration with a 'score' key.
        n (int, optional): The factor to multiply each score by before normalization. Defaults to 1.
    
    Returns:
        list of dict: The input list of dictionaries with normalized 'score' values.
    """
    
    # Extract scores and multiply by n
    scores = np.array([configuration['score'] for configuration in config]) ** n
    
    # Normalize scores to sum to 1
    normalized_scores = scores / np.sum(scores)
    
    # Update the scores in the original list of dictionaries
    for configuration, new_score in zip(config, normalized_scores):
        configuration['score'] = new_score
    
    return config

def extract_generated_texts(tokenizer, input_ids_0: torch.Tensor, output: torch.Tensor) -> List[str]:
    """
    Extract generated text from the model's output, excluding the input portion and any left-side padding.

    :param tokenizer: The tokenizer used, which must have a pad_token_id attribute.
    :param input_ids_0: Token IDs input to the model, shaped (batch_size, sequence_length).
                        Input may contain left-side padding.
    :param output: Model output token IDs, shaped (batch_size, output_sequence_length).
                Output sequence contains both the input sequence and the generated response.
    :return: A list of strings, where each string is the generated text for the corresponding batch.

    Function logic:
    - For each sample, find the non-pad portion in input_ids_0.
    - Search for a matching sequence in the output that corresponds to the non-pad portion.
    - Extract from the end of the matched sequence in the output to the end of the output as the response.
    - Decode the token IDs of the response into text using the tokenizer.
    """
    pad_token_id = tokenizer.pad_token_id
    generated_texts = []

    for i in range(output.shape[0]):
        # Find the index of the first non-pad token in input_ids_0
        non_pad_indices = (input_ids_0[i] != pad_token_id).nonzero().squeeze()
        
        if non_pad_indices.dim() == 0:
            non_pad_indices = non_pad_indices.unsqueeze(0)

        first_non_pad_index = non_pad_indices[0].item() if non_pad_indices.numel() > 0 else -1

        if first_non_pad_index == -1:
            raise ValueError("No non-pad tokens found in the input for batch index {}".format(i))

        # Construct the input_ids tensor of the non-pad portion for the current sample
        input_ids_non_pad = input_ids_0[i, first_non_pad_index:]

        found_match = False
        for pos in range(output.shape[1]):
            if pos + input_ids_non_pad.shape[0] <= output.shape[1]:
                if torch.equal(output[i, pos:pos+input_ids_non_pad.shape[0]], input_ids_non_pad):
                    found_match = True
                    response_start_index = pos + input_ids_non_pad.shape[0]
                    break

        if not found_match:
            raise ValueError(f"No matching sequence found in the output for batch index {i}")

        response_ids = output[i, response_start_index:]

        decoded_text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        generated_texts.append(decoded_text)

    return generated_texts

def update_scores(config, norm_type):
    """
    This function updates each dictionary in a list by different strategies based on the norm_type value.
    - 'average': Sets all scores to 1.
    - 'score': Leaves the "score" values unchanged.
    
    If the norm_type is not one of the specified values, an error is raised.

    Parameters:
    - config (list of dict): A list of dictionaries, each containing the fields "score" and "ece".
    - norm_type (str): The type of normalization to apply ('average' or 'score').

    Returns:
    - The updated list of dictionaries according to the specified normalization type.
    
    Raises:
    - ValueError: If norm_type is not one of the specified values.
    """
    if norm_type == 'average':
        for item in config:
            item["score"] = 1
    elif norm_type == 'score':
        pass
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}. Expected 'average' or 'score'.")

    return config

def load_yaml_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    try:
        config_api_server = config['CONFIG_API_SERVER']
        norm_type_api_server = config['NORM_TYPE_API_SERVER']
        threshold_api_server = config['THRESHOLD_API_SERVER']
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")

    return config_api_server, norm_type_api_server, threshold_api_server

# init RAY
ray.init()
