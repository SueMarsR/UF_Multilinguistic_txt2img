from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载模型和分词器
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# 输入文本
input_text = "The quick brown fox jumps over the lazy dog."

# 构建模型输入
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 生成扩写文本
output = model.generate(input_ids=input_ids, max_length=50, do_sample=True)

# 解码输出文本
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印输出文本
print(output_text)
