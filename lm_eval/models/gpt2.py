import transformers
import torch
import torch.nn.functional as F
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm


class GPT2LM(LM):
    MAX_GEN_TOKS = 256

    def __init__(self, device=None, pretrained='gpt2', dropout=None):
        super().__init__()
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        config = transformers.AutoConfig.from_pretrained(pretrained)
        config_d = config.to_dict()
        # if dropout is not None:
        #     for k in config_d.keys():
        #         if "drop" in k:
        #             config_d[k] = float(dropout)
        # config = config.from_dict(config_d)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(pretrained, config=config).eval()

        # pretrained tokenizer for neo is broken for now so just hardcoding this to gpt2
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = "<|endoftext|>"
        try:
            self.max_length = self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparantly
            self.max_length = self.model.config.max_position_embeddings

        assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]

    @classmethod
    def create_from_arg_string(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(**args)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [50256]
            else:
                context_enc = self.tokenizer.encode(context)

            continuation_enc = self.tokenizer.encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(self, requests):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []
        with torch.no_grad():
            # TODO: vectorize properly
            # TODO: automatic batch size detection for vectorization

            def _collate(x):
                toks = x[1] + x[2]
                return (len(toks), tuple(toks))
            
            reord = utils.Reorderer(requests, _collate)
            for cache_key, context_enc, continuation_enc in tqdm(reord.get_reordered()):
                # when too long to fit in context, truncate from the left
                inp = torch.tensor([(context_enc + continuation_enc)[-self.max_length:]], dtype=torch.long).to(self.device)
                ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - self.max_length)

                cont_toks = inp[:, ctxlen:]  # [batch, seq]
                logits = F.log_softmax(self.model(inp)[0][:, :, :50257], dim=-1)[:, ctxlen - 1:-1]  # [batch, seq, vocab]

                greedy_tokens = logits.argmax(dim=-1)
                max_equal = (greedy_tokens == cont_toks).all()

                last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1) # [batch, seq]

                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return reord.get_original(res)
    
    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles untils that are 
        # multiple tokens or that span multiple tokens correctly
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])
        
        reord = utils.Reorderer(requests, _collate)

        for context, until in tqdm(reord.get_reordered()):
            if isinstance(until, str): until = [until]

            context_enc = torch.tensor([self.tokenizer.encode(context)[self.MAX_GEN_TOKS - self.max_length:]]).to(self.device)

            primary_until, = self.tokenizer.encode(until[0])

            cont = self.model.generate(
                context_enc,
                max_length=context_enc.shape[1] + self.MAX_GEN_TOKS,
                eos_token_id=primary_until,
                do_sample=False
            )

            s = self.tokenizer.decode(cont[0].tolist()[context_enc.shape[1]:])

            for term in until:
                s = s.split(term)[0]
            
            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            
            res.append(s)
        
        return reord.get_original(res)
