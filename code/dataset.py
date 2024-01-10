import json
import os
import re
import glob
import torch as th


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_algebra_examples(dataroot, test_num):
    path = os.path.join(dataroot, f"test_{test_num}.jsonl")
    examples = read_jsonl(path)
    # filtered_examples = []

    print(f"{len(examples)} test_{test_num} examples")

    return examples


def get_examples(dataroot, if_eval=False):
    # print(dataroot)
    all_filenames = glob.glob(dataroot)
    # print(len(all_filenames))
    samples_raw = []
    for fname in all_filenames:
        with open(fname, 'r') as fp:
            try:
                problem_data = json.load(fp)
            except Exception as e:
                print(f"Error loading JSON from {fname}", e)
                raise e
        curr_sample_raw = {'question': 'Problem:\n' + problem_data['problem'],
                           'answer': '\nSolution:\n' + problem_data['solution']}
        for e in curr_sample_raw:
            assert e
        # if len(curr_sample_raw['question']) < 2048:
        samples_raw.append(curr_sample_raw)
    print(f'{dataroot}', len(samples_raw))
    return samples_raw


class MATHCHATFILEDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, dataroot, loss_on_prefix=True):
        self.dataroot = dataroot
        self.samples = None
        self.initialize()

        self.qns_ = [ex["question"] for ex in self.samples]
        self.ans_ = [ex["answer"] for ex in self.samples]
        self.qns = tokenizer(self.qns_, padding=False)
        self.ans = tokenizer(self.ans_, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.samples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def initialize(self):
        """
        Set up self.samples by loading from the dataroot
        """
        all_filenames = glob.glob(self.dataroot)
        samples_raw = []
        for fname in all_filenames:
            examples = read_jsonl(fname)
            for ex in examples:
                for i, generated_answer in enumerate(ex['generated_answer']):
                    output = remove_boxed(last_boxed_only_string(generated_answer))
                    label = remove_boxed(last_boxed_only_string(ex["answer"]))
                    if is_equiv(output, label):
                        answer = '\n' + generated_answer
                        curr_sample_raw = {'question': ex['question'],
                                           'answer': answer}
                        for e in curr_sample_raw:
                            assert e
                        if len(curr_sample_raw['answer'] + curr_sample_raw['question']) < 1200:
                            samples_raw.append(curr_sample_raw)

        self.samples = samples_raw
        del samples_raw

        print(f"{self.__class__.__name__}: Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
                ([int(self.loss_on_prefix)] * len(qn_tokens))
                + ([1] * len(ans_tokens))
                + ([0] * len(pad_tokens))
        )
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        with open("/mnt/public03/usr/liyiwei/math_gpt4/log_beta.txt","a")as f:
            #f.write(str(beta[0].item()))
            f.write("\nQuestion!!"+self.qns_[idx]+"Questionend\n")
            f.close()
        return dict(input_ids=tokens, attention_mask=mask, labels=tokens,ques=self.qns_[idx])


class MATHCHATFILEZFDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, dataroot, loss_on_prefix=True,stage=1):
        self.dataroot = dataroot
        self.samples = None
        self.stage=stage
        if stage==1:
            self.pd = False
        else:
            self.pd = True
        self.initialize()

        self.qns = [ex["question"] for ex in self.samples]
        self.ans = [ex["answer"] for ex in self.samples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.samples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def initialize(self):
        """
        Set up self.samples by loading from the dataroot
        """
        all_filenames = glob.glob(self.dataroot)
        samples_raw = []
        for fname in all_filenames:
            examples = read_jsonl(fname)
            for ex in examples:
                for i, generated_answer in enumerate(ex['generated_answer']):
                    output = remove_boxed(last_boxed_only_string(generated_answer))
                    label = remove_boxed(last_boxed_only_string(ex["answer"]))
                    if is_equiv(output, label) == self.pd:
                        answer = '\n' + generated_answer
                        curr_sample_raw = {'question': ex['question'],
                                           'answer': answer}
                        for e in curr_sample_raw:
                            assert e
                        if len(curr_sample_raw['answer'] + curr_sample_raw['question']) < 1600:
                            samples_raw.append(curr_sample_raw)

        self.samples = samples_raw
        del samples_raw

        print(f"{self.__class__.__name__}: Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
                ([int(self.loss_on_prefix)] * len(qn_tokens))
                + ([1] * len(ans_tokens))
                + ([0] * len(pad_tokens))
        )
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask, labels=tokens)


class MATHCHATNTDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, dataroot, loss_on_prefix=True):
        self.dataroot = dataroot
        self.samples = None
        self.initialize()

        self.qns = [ex["question"] for ex in self.samples]
        self.ans = [ex["answer"] for ex in self.samples]
        self.scores = [ex["score"] for ex in self.samples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.samples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def initialize(self):
        """
        Set up self.samples by loading from the dataroot
        """
        all_filenames = glob.glob(self.dataroot)
        samples_raw = []
        for fname in all_filenames:
            examples = read_jsonl(fname)
            for ex in examples:
                curr_sample_raw = {'question': ex['question'],
                                   'answer': ex["answer"],
                                   'score': 1}
                if len(curr_sample_raw['answer'] + curr_sample_raw['question']) < 1600:
                    samples_raw.append(curr_sample_raw)
                for i, generated_answer in enumerate(ex['generated_answer']):
                    output = remove_boxed(last_boxed_only_string(generated_answer))
                    label = remove_boxed(last_boxed_only_string(ex["answer"]))
                    answer = '\n' + generated_answer
                    if is_equiv(output, label):
                        curr_sample_raw = {'question': ex['question'],
                                           'answer': answer,
                                           'score': 1}
                    else:
                        curr_sample_raw = {'question': ex['question'],
                                           'answer': answer,
                                           'score': -0.05}
                    for e in curr_sample_raw:
                        assert e
                    if len(curr_sample_raw['answer'] + curr_sample_raw['question']) < 1600:
                        samples_raw.append(curr_sample_raw)

        self.samples = samples_raw
        del samples_raw

        print(f"{self.__class__.__name__}: Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        scores = self.scores[idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
                ([int(self.loss_on_prefix)] * len(qn_tokens))
                + ([1] * len(ans_tokens))
                + ([0] * len(pad_tokens))
        )
        scores = th.tensor(scores)
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask, labels=tokens, scores=scores)


class MATHCHATCLDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, dataroot, loss_on_prefix=True):
        self.dataroot = dataroot
        self.samples = None
        self.initialize()

        self.qns = [ex[0]["question"] for ex in self.samples]
        self.ans = []
        for ex in self.samples:
            for sample in ex:
                self.ans.append(sample["answer"])
        self.scores = []
        for ex in self.samples:
            for sample in ex:
                self.scores.append(sample["score"])
        # self.ans = [ex["answer"] for ex in self.samples]
        # self.scores = [ex["score"] for ex in self.samples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + max([len(self.ans["input_ids"][i*8 + j]) for j in range(8)])
                for i in range(len(self.samples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def initialize(self):
        """
        Set up self.samples by loading from the dataroot
        """
        all_filenames = glob.glob(self.dataroot)
        samples_raw = []
        for fname in all_filenames:
            examples = read_jsonl(fname)
            for ex in examples:
                session = []
                flag = 0
                pad = None
                curr_sample_raw = {'question': ex['question'],
                                   'answer': ex["answer"],
                                   'score': 1}
                if len(curr_sample_raw['answer'] + curr_sample_raw['question']) < 1600:
                    session.append(curr_sample_raw)
                    pad = curr_sample_raw
                for i, generated_answer in enumerate(ex['generated_answer']):
                    output = remove_boxed(last_boxed_only_string(generated_answer))
                    label = remove_boxed(last_boxed_only_string(ex["answer"]))
                    answer = '\n' + generated_answer
                    if is_equiv(output, label):
                        curr_sample_raw = {'question': ex['question'],
                                           'answer': answer,
                                           'score': 1}
                        if len(curr_sample_raw['answer'] + curr_sample_raw['question']) < 1600:
                            session.append(curr_sample_raw)
                            if not pad:
                                pad = curr_sample_raw
                    else:
                        if flag == 1:
                            curr_sample_raw = {'question': ex['question'],
                                               'answer': answer,
                                               'score': 0}
                            if len(curr_sample_raw['answer'] + curr_sample_raw['question']) < 1600:
                                session.append(curr_sample_raw)
                        else:
                            flag = 1
                if not session:
                    continue
                if len(session) > 8:
                    session = session[:-1]
                elif len(session) < 8:
                    if not pad:
                        continue
                        # pad = session[-1]
                    for i in range(8 - len(session)):
                        session.append(pad)
                # print(len(session))
                assert len(session) == 8
                samples_raw.append(session)
                del session

        self.samples = samples_raw
        del samples_raw

        print(f"{self.__class__.__name__}: Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        qn_token = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx*8: (idx+1)*8]
        scores = self.scores[idx*8: (idx+1)*8]
        mask = []
        tokens = []
        question_mask = []
        for ans_token in ans_tokens:
            pad_token = [0] * (self.max_len - len(qn_token) - len(ans_token))
            tokens.append(qn_token + ans_token + pad_token)
            mask.append((
                    ([int(self.loss_on_prefix)] * len(qn_token))
                    + ([1] * len(ans_token))
                    + ([0] * len(pad_token))
            ))
            question_mask.append((
                    ([int(self.loss_on_prefix)] * len(qn_token))
                    + ([0] * len(ans_token))
                    + ([0] * len(pad_token))
            ))
        scores = th.tensor(scores)
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask, labels=tokens, scores=scores, question_mask=question_mask)


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string




def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
