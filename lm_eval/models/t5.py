import transformers
import torch
import torch.nn.functional as F
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm


class T5LM(LM):
    MAX_GEN_TOKS = 256

    def __init__(self, device=None, pretrained='gpt2'):
        super().__init__()
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained)
        self.model.parallelize()
        self.model.eval()

        self.tokenizer = transformers.T5TokenizerFast.from_pretrained('t5-base')
        self.max_length = 512
        # self.tokenizer.pad_token = "<|endoftext|>"
        # try:
        #     self.max_length = self.gpt2.config.n_ctx
        # except AttributeError:
        #     # gptneoconfig doesn't have n_ctx apparantly
        #     self.max_length = self.gpt2.config.max_position_embeddings
        #
        # assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]

    @classmethod
    def create_from_arg_string(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(device=args.get("device", None), pretrained=args.get("pretrained", "t5-small"))

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                raise ValueError("Yuval - I didn't handle this case")
            else:
                context_enc = self.tokenizer.encode(context + "<extra_id_0>.")

            continuation_enc = self.tokenizer.encode("<extra_id_0> " + continuation, add_special_tokens=False)

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

                context_toks = inp[:, :ctxlen]
                continuation_toks = inp[:, ctxlen:]  # [batch, seq]
                logits = F.log_softmax(self.model(input_ids=context_toks, labels=continuation_toks)[1], dim=-1) # [batch, cont, vocab]

                greedy_tokens = logits.argmax(dim=-1)
                max_equal = (greedy_tokens == continuation_toks).all()

                last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                logits = torch.gather(logits, 2, continuation_toks.unsqueeze(-1)).squeeze(-1) # [batch, seq]

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

            context_enc = torch.tensor([self.tokenizer.encode(context.strip() + "<extra_id_0>.")[- self.max_length:]]).to(self.device)

            primary_until, = self.tokenizer.encode(until[0])

            cont = self.model.generate(
                context_enc,
                max_length=self.MAX_GEN_TOKS,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<extra_id_1>"),
                do_sample=False
            )

            s = self.tokenizer.decode(cont[0].tolist()[2:-1])
            if "<extra_id_1>" in s:
                s = s[:s.index("<extra_id_1>")]
            if "</s>" in s:
                s = s[:s.index("</s>")]
            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            
            res.append(s)
        
        return reord.get_original(res)
