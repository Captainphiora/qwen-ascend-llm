import numpy as np
import os
import time
import gc
from transformers import AutoTokenizer
from enum import Enum
from threading import Lock
from utils.session import Session
from config import InferenceConfig
from tqdm import trange, tqdm
import torch

try:
    import torch_npu
    HAS_TORCH_NPU = True
except (ImportError, RuntimeError):
    HAS_TORCH_NPU = False



# Inference类 负责tokenizer管理、对话模板、token采样和生成循环：
class Inference:
    def __init__(self, config: InferenceConfig) -> None:
        self.max_input_length = config.max_input_length
        self.max_output_length = config.max_output_length
        # self.tokenizer=Tokenizer(config.tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_dir, trust_remote_code=True
        )
        self.sampling_method = config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature = config.temperature
        self.system_prompt = config.system_prompt
        self.session = Session.fromConfig(config)
        self.session_type = config.session_type
        if config.device_str == "cpu":
            self.torch_device = torch.device("cpu")
        elif config.device_str == "cuda":
            self.torch_device = torch.device("cuda")
        elif config.device_str == "npu":
            self.torch_device = torch.device("npu")
        else:
            raise Exception(f"unsport device {config.device_str}")
        # self.prompt=config.prompt
        self.kv_cache_length = config.kv_cache_length
        self.state: dict = {"code":200,"isEnd":False,"message":""}
        self.use_npu_sampling = (
            HAS_TORCH_NPU
            and config.session_type == "acl"
            and os.environ.get("USE_NPU_SAMPLING", "0") == "1"
        )
        if HAS_TORCH_NPU and config.session_type == "acl":
            self.npu_sampling_device = f"npu:{config.device_id}"
        if self.use_npu_sampling:
            self.session.model._skip_logits_d2h = True
            vocab_size = config.vocab_size
            self._npu_logits_buffer = torch.empty(
                1, vocab_size, dtype=torch.float32,
                device=self.npu_sampling_device
            )
        self.reset()
        self.lock = Lock()
        self.first = True
        # self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5}
        print("[INFO] init success")

    @staticmethod
    def _get_last_logits(logits):
        """从 session.run() 的返回中提取最后一个 token 的 logits。
        logits 可以是 numpy array 或 device dict。"""
        if isinstance(logits, dict):
            return logits
        return logits[0][-1:]


    def generate_cache(self, prompt: str):
        """
        生成kv-cache
        Args:
            prompt (str): 提示词

        Returns:
            返回下一个token与logits
        """
        if len(prompt) == 0 :
            return
        self.first = False
        input_ids = np.asarray(
            self.tokenizer.encode(prompt), dtype=np.int64
        ).reshape(1,-1)
        logits = self.session.run(input_ids)[0]
        next_token = self.sample_logits(
            self._get_last_logits(logits),
            self.sampling_method,
            self.sampling_value,
            self.temperature
        ) 
        return next_token, logits

    def sample_logits(
        self,
        logits,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        对logits做采样，得到下一个token。
        logits 可以是:
          - np.ndarray [1, vocab_size] (CPU 路径)
          - dict {'device_ptr':..., 'nbytes':..., 'shape':..., 'dtype':...} (NPU zero-copy 路径)

        Returns:
            np.ndarray: 下一个 token id
        """
        if self.use_npu_sampling:
            return self._sample_logits_npu(
                logits, sampling_method, sampling_value, temperature
            )
        return self._sample_logits_cpu(
            logits, sampling_method, sampling_value, temperature
        )

    def _sample_logits_npu(
        self,
        logits,
        sampling_method: str,
        sampling_value: float,
        temperature: float,
    ) -> np.ndarray:
        """
        NPU ATB 采样 (zero-copy): logits 留在 device，D2D 拷贝到 torch tensor 后采样。
        仅回传 1 个 token id (8 bytes) 到 Host。
        """
        import acl
        ACL_MEMCPY_DEVICE_TO_DEVICE = 3

        if isinstance(logits, dict):
            device_ptr = logits['device_ptr']
            nbytes = logits['nbytes']
            vocab_size = logits['shape'][-1]
        else:
            raise ValueError("NPU sampling expects device dict, got numpy")

        if temperature == 0 or sampling_method == "greedy":
            logits_t = self._npu_logits_buffer[:, :vocab_size]
            nbytes_dst = logits_t.nelement() * logits_t.element_size()
            ret = acl.rt.memcpy(
                logits_t.data_ptr(), nbytes_dst,
                device_ptr, nbytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE
            )
            idx = logits_t.argmax(dim=-1)
            return idx.cpu().numpy().flatten().astype(np.int64)

        logits_t = self._npu_logits_buffer[:, :vocab_size]
        nbytes_dst = logits_t.nelement() * logits_t.element_size()
        ret = acl.rt.memcpy(
            logits_t.data_ptr(), nbytes_dst,
            device_ptr, nbytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE
        )
        logits_t = (logits_t / temperature).half()

        if sampling_method == "top_k":
            top_k = int(sampling_value)
            top_k_t = torch.tensor(
                [min(top_k, 1024)], dtype=torch.int32, device=self.npu_sampling_device
            )
            top_p_t = torch.tensor(
                [1.0], dtype=torch.float16, device=self.npu_sampling_device
            )
            q = torch.rand(
                1, vocab_size,
                dtype=torch.float32, device=self.npu_sampling_device
            )
            idx, _ = torch_npu.npu_top_k_top_p_sample(
                logits_t, top_k_t, top_p_t, q
            )
        elif sampling_method == "top_p":
            top_k_t = torch.tensor(
                [100], dtype=torch.int32, device=self.npu_sampling_device
            )
            top_p_t = torch.tensor(
                [sampling_value], dtype=torch.float16, device=self.npu_sampling_device
            )
            q = torch.rand(
                1, vocab_size,
                dtype=torch.float32, device=self.npu_sampling_device
            )
            idx, _ = torch_npu.npu_top_k_top_p_sample(
                logits_t, top_k_t, top_p_t, q
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return idx.cpu().numpy().flatten()

    def _sample_logits_cpu(
        self,
        logits: np.ndarray,
        sampling_method: str,
        sampling_value: float,
        temperature: float,
    ) -> np.ndarray:
        """
        CPU numpy 采样 (fallback)。
        """
        if temperature == 0 or sampling_method == "greedy":
            if logits.dtype != np.float32:
                logits = logits.astype(np.float32)
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None
            logits = logits[0].astype(np.float32)
            logits /= temperature
            logits -= np.max(logits)

            if sampling_method == "top_k":
                top_k = int(sampling_value)
                top_indices = np.argpartition(logits, -top_k)[-top_k:]
                top_logits = logits[top_indices]
                top_logits -= np.max(top_logits)
                top_probs = np.exp(top_logits)
                top_probs /= np.sum(top_probs)
                next_token = np.array([np.random.choice(top_indices, p=top_probs)])

            elif sampling_method == "top_p":
                p = sampling_value
                k_candidate = min(100, logits.shape[-1])
                top_k_indices = np.argpartition(logits, -k_candidate)[-k_candidate:]
                top_k_logits = logits[top_k_indices]
                sorted_order = np.argsort(top_k_logits)[::-1]
                sorted_logits = top_k_logits[sorted_order]
                sorted_indices = top_k_indices[sorted_order]
                sorted_logits -= sorted_logits[0]
                sorted_probs = np.exp(sorted_logits)
                sorted_probs /= np.sum(sorted_probs)
                cumulative_probs = np.cumsum(sorted_probs)
                if cumulative_probs[-1] < p:
                    k_candidate = min(1000, logits.shape[-1])
                    top_k_indices = np.argpartition(logits, -k_candidate)[-k_candidate:]
                    top_k_logits = logits[top_k_indices]
                    sorted_order = np.argsort(top_k_logits)[::-1]
                    sorted_logits = top_k_logits[sorted_order]
                    sorted_indices = top_k_indices[sorted_order]
                    sorted_logits -= sorted_logits[0]
                    sorted_probs = np.exp(sorted_logits)
                    sorted_probs /= np.sum(sorted_probs)
                    cumulative_probs = np.cumsum(sorted_probs)
                cutoff = int(np.searchsorted(cumulative_probs, p)) + 1
                top_indices = sorted_indices[:cutoff]
                top_probs = sorted_probs[:cutoff]
                top_probs /= np.sum(top_probs)
                next_token = np.array([np.random.choice(top_indices, p=top_probs)])
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token

    # 流式推理
    def stream_predict(
        self,
        prompt,
        history=None,
        sampling_config: dict = {},
        system_prompt: str = None,
        max_new_tokens: int = 1024,
        do_speed_test: bool = False,
        show_progress: bool = False,
    ):
        if history is None:
            history = [] 
        if system_prompt is None:
            system_prompt = self.system_prompt
        sampling_value = sampling_config.get("sampling_value", self.sampling_value)
        temperature = sampling_config.get("temperature", self.temperature)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # print("prompt: ", prompt)
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""   
        if prompt == "":
            return
        for (use_msg, bot_msg) in history:
            messages.append({"role": "user", "content": use_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": prompt})
        # print("history: ", history)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # ===== 正式推理前，打印真正喂给模型的输入与本次采样参数 =====
        print("\n[DEBUG] ---- 本次推理实际输入(stream_predict) ----")
        print("[DEBUG] messages: {}".format(messages))
        print("[DEBUG] 应用 chat_template 后的完整 prompt:\n{!r}".format(text))
        print("[DEBUG] 本次采样: sampling_method={}, sampling_value={}, temperature={}, greedy={}".format(
            self.sampling_method, sampling_value, temperature,
            (temperature == 0) or (self.sampling_method == "greedy")))
        print("[DEBUG] ---------------------------")
        # acl和onnx推理tokenizer处理
        if self.session_type in ["onnx", "acl"]:
            input_ids = self.tokenizer(
                [text], return_tensors="np"
            )["input_ids"].astype(np.int64).reshape(1, -1)
        elif self.session_type == "pytorch":
            input_ids = self.tokenizer(
                [text], return_tensors="pt"
            )["input_ids"].to(torch.long).reshape(1, -1).to(self.torch_device)
        else:
            raise Exception(f"unknown session_type {self.session_type}")
        input_ids = input_ids[:, -self.max_input_length:]
        # print("input_ids shape: ", input_ids.shape)
        self.first = False
        ids_list = []
        text_length = 0
        input_length = input_ids.shape[1]
        if do_speed_test:
            first_token_start = time.time()
            first_token_latency = 0
            decode_speed = 0
        max_output_len = self.max_output_length - input_length
        max_output_len = min(max_output_len, max_new_tokens)
        if show_progress:
            temp_list = trange(max_output_len, desc="decode")
        else:
            temp_list = range(max_output_len)
        prefill_show_progress = False
        decode_speed, totol_speed = 0.0, 0.0
        for i in temp_list:
            if i == 0:
                if show_progress:
                    prefill_show_progress = True
                # reset counter
                # 清空device kvcache
                self.session.reset()
            else:
                prefill_show_progress = False
            # 前向推理
            logits = self.session.run(
                input_ids,
                show_progress=prefill_show_progress,
            )
            # 采样下一个token
            input_ids = self.sample_logits(
                self._get_last_logits(logits),
                self.sampling_method,
                sampling_value,
                temperature
            )
            input_ids = input_ids.reshape(1, -1)
            if do_speed_test and i == 0:
                decode_token_start = time.time()
                first_token_latency = decode_token_start - first_token_start
            with self.lock:
                # early stop
                if input_ids[0] == self.tokenizer.eos_token_id:
                    self.state['message'],self.state['isEnd'] = self.tokenizer.decode(ids_list),True
                    break
                ids_list.append(input_ids[0].item())
                text_out = self.tokenizer.decode(ids_list)
                # stop_word = is_stop_word_or_prefix(text_out, ["[|Human|]", "[|AI|]"])
                self.state['message'] = text_out
                new_text = text_out[text_length: ]
                if do_speed_test and i > 0:
                    now_time = time.time()
                    decode_duration = now_time - decode_token_start
                    total_duration = now_time - first_token_start
                    decode_speed = (len(ids_list) - 1) / decode_duration
                    totol_speed = (input_length + len(ids_list)) / total_duration
                if b"\xef\xbf\xbd" in new_text.encode():
                    continue
                if len(new_text) > 0:
                    if do_speed_test:
                        yield new_text, first_token_latency, decode_speed, totol_speed
                    else:
                        yield new_text
                    text_length = len(text_out)
        with self.lock:
            self.state['isEnd'] = True
    
    def predict(
        self,
        prompt,
        history=None,
        sampling_config: dict = {},
        system_prompt: str = None,
        max_new_tokens: int = 1024,
        show_progress: bool = False,
    ):
        if history is None:
            history = []
        if system_prompt is None:
            system_prompt = self.system_prompt
        sampling_value = sampling_config.get("sampling_value", self.sampling_value)
        temperature = sampling_config.get("temperature", self.temperature)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # print("prompt: ", prompt)
        with self.lock:
            self.state['isEnd'], self.state['message'] = False,""
        if prompt == "":
            return    
        for (use_msg, bot_msg) in history:
            messages.append({"role": "user", "content": use_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": prompt})
        # print("history: ", history)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if self.session_type in ["onnx", "acl"]:
            input_ids = self.tokenizer(
                [text], return_tensors="np"
            )["input_ids"].astype(np.int64).reshape(1, -1)
        elif self.session_type == "pytorch":
            input_ids = self.tokenizer(
                [text], return_tensors="pt"
            )["input_ids"].to(torch.long).reshape(1, -1).to(self.torch_device)
        else:
            raise Exception(f"unknown session_type {self.session_type}")
        input_ids = input_ids[:, -self.max_input_length:]
        self.first = False
        ids_list = []
        # text_length = 0
        input_length = input_ids.shape[1]
        # start = time.time()
        # first_token_latency = 0
        # decode_speed = 0
        max_output_len = self.max_output_length - input_length
        max_output_len = min(max_output_len, max_new_tokens)
        if show_progress:
            temp_list = trange(max_output_len, desc="decode")
        else:
            temp_list = range(max_output_len)
        prefill_show_progress = False
        for i in temp_list:
            if i == 0:
                if show_progress:
                    prefill_show_progress = True
                # reset counter
                self.session.reset()
            else:
                prefill_show_progress = False
            logits = self.session.run(
                input_ids,
                show_progress=prefill_show_progress
            )
            input_ids = self.sample_logits(
                self._get_last_logits(logits),
                self.sampling_method,
                sampling_value,
                temperature
            )
            input_ids = input_ids.reshape(1, -1)
            # if i == 0:
            #     first_token_latency = time.time() - start
            with self.lock:
                # early stop
                if input_ids[0] == self.tokenizer.eos_token_id:
                    self.state['message'],self.state['isEnd'] = self.tokenizer.decode(ids_list),True
                    break
                ids_list.append(input_ids[0].item())
                # text_out = self.tokenizer.decode(ids_list)
                # stop_word = is_stop_word_or_prefix(text_out, ["[|Human|]", "[|AI|]"])
                # self.state['message'] = text_out
                # decode_speed =
        with self.lock:
            self.state['isEnd'] = True
        text_out = self.tokenizer.decode(ids_list)
        return text_out
    
    def generate(
        self,
        input_ids,
        sampling_config: dict = {},
        max_new_tokens: int = 1024,
        show_progress: bool = False,
    ):
        sampling_value = sampling_config.get("sampling_value", self.sampling_value)
        temperature = sampling_config.get("temperature", self.temperature)
        self.first = False
        ids_list = []
        input_ids = input_ids[:, -self.max_input_length:]
        input_length = input_ids.shape[1]
        max_output_len = self.max_output_length - input_length
        max_output_len = min(max_output_len, max_new_tokens)
        if show_progress:
            temp_list = trange(max_output_len, desc="decode")
        else:
            temp_list = range(max_output_len)
        prefill_show_progress = False
        for i in temp_list:
            if i == 0:
                if show_progress:
                    prefill_show_progress = True
                # reset counter
                self.session.reset()
            else:
                prefill_show_progress = False
            logits = self.session.run(
                input_ids,
                show_progress=prefill_show_progress
            )
            input_ids = self.sample_logits(
                self._get_last_logits(logits),
                self.sampling_method,
                sampling_value,
                temperature
            )
            input_ids = input_ids.reshape(1, -1)
            with self.lock:
                # early stop
                if input_ids[0] == self.tokenizer.eos_token_id:
                    self.state['message'],self.state['isEnd'] = self.tokenizer.decode(ids_list),True
                    break
                ids_list.append(input_ids[0].item())
                text_out = self.tokenizer.decode(ids_list)
                # print("Debug: ", text_out)
                # stop_word = is_stop_word_or_prefix(text_out, ["[|Human|]", "[|AI|]"])
                self.state['message'] = text_out
        with self.lock:
            self.state['isEnd'] = True 
        text_out = self.tokenizer.decode(ids_list)
        return text_out

    def reset(self):
        self.first = True
        self.session.run_times = 0
        self.session.reset()
        # self.generate_cache(self.prompt)


    def getState(self):
        with self.lock:
            return self.state.copy()

# def preprocess(text:str) -> str:
#     # 将输入转换为指定格式
#     return f"<|user|>\n{text}</s>\n<|assistant|>"
#     
# 
# def is_stop_word_or_prefix(s: str, stop_words: list) -> int:
#     for stop_word in stop_words:
#         if s.endswith(stop_word):
#             return stop_word
#     return ""
# 
