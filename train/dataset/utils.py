import re
import os


def pre_question(question, max_ques_words):
    question = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            question.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
    )
    question = question.rstrip(" ")

    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])

    return question


def pre_caption(caption, max_words):
    caption = (
        re.sub(
            # r"([,.'!?\"()*#:;~])",
            r"([.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
        .replace("<person>", "person")
    )

    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if caption_words[0].lower() == "shein":
        caption_words = caption_words[1:]
    if max_words is not None and len(caption_words) > max_words:
        caption_words = caption_words[:max_words]
    caption = " ".join(caption_words)
    # caption不能为空
    if caption == "":
        caption = "a cloth"
    # if not caption.isascii():
    #     # caption = 'a product'
    # print(caption)
    return caption


def chunk(iterable, chunk_size, drop_last=False):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    drop_last = drop_last and len(ret) != chunk_size
    if not drop_last:
        yield ret


def filter_text(text):
    from_list = ["！", "＂", "＃", "＄", "％", "＆", "＇", "（", "）", "＊", "＋", "，", "－", "．", "／", "：", "；", "＜", "＝", "＞", "？", "＠", "［", "］", "＾", "＿", "｀", "｛", "｜", "｝", "～", "￮", "【", "】", "\u00a0", "。", "、"]
    to_list = ["!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", ".", "[", "]", " ", ".", ","]
    trans_dict = str.maketrans({f: h for f, h in zip(from_list, to_list)})

    def text_preprocess(text):
        """ "处理掉html标签，在右侧标签处加一些句号"""
        text = text.translate(trans_dict)
        text = re.sub("<img .*?>", "", text)
        text = re.sub("<p *>|<li *>|<ul *>|\-|#|•|&\s?amp;|\*|&\s?nbsp;|<--sep--\s*\d+\s*/>|=|&gt;|●", " ", text)
        text = re.sub("</p>|</li>|</ul>", ". ", text)
        text = re.sub("\.+(\.|\s)+", ". ", text)  # 多个句号夹杂空格
        text = re.sub("^\s*\.+(\.|\s)*", "", text)  # 开头的句号
        text = re.sub("\s+", " ", text)  # 多个连续空格

        return text.strip()

    return text_preprocess(text)
