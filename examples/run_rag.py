# To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
# from transformers import RagRetriever
#
# retriever = RagRetriever.from_pretrained(
#     "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
# )

import torch
# >> >  # To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
# >> > from transformers import RagRetriever
#
# >> > dataset = (
#     ......
#     ...)  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index
# >> > retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", indexed_dataset=dataset)
#
# >> >  # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
# >> > from transformers import RagRetriever
#
# >> > dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*
# >> > index_path = "path/to/my/index.faiss"  # faiss index saved via *dataset.get_index("embeddings").save(...)*
# >> > retriever = RagRetriever.from_pretrained(
#     ...
# "facebook/dpr-ctx_encoder-single-nq-base",
# ...
# index_name = "custom",
# ...
# passages_path = dataset_path,
# ...
# index_path = index_path,
# ... )
#
# >> >  # To load the legacy index built originally for Rag's paper
# >> > from transformers import RagRetriever
#
# >> > retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", index_name="legacy")
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
)

# initialize with RagRetriever to do everything in one forward call
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
input_ids = inputs["input_ids"]
labels = targets["input_ids"]
outputs = model(input_ids=input_ids, labels=labels)

# or use retriever separately
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
# 1. Encode
question_hidden_states = model.question_encoder(input_ids)[0]
# 2. Retrieve
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
doc_scores = torch.bmm(
    question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
).squeeze(1)

# 3. Forward to generator
outputs = model(
    context_input_ids=docs_dict["context_input_ids"],
    context_attention_mask=docs_dict["context_attention_mask"],
    doc_scores=doc_scores,
    decoder_input_ids=labels,
)

# or directly generate
generated = model.generate(
    context_input_ids=docs_dict["context_input_ids"],
    context_attention_mask=docs_dict["context_attention_mask"],
    doc_scores=doc_scores,
)
generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
