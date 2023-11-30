import pinecone
import time
import pandas as pd
from sentence_transformers import SentenceTransformer

class preprocess():
    def __init__(self):
        self.pinecone_api_key = "2b414101-2b40-4c52-a9ed-92223cbff6f9"
        self.pinecone_env = "gcp-starter"
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_docs(self, docs):
        out = self.encoder.encode(docs)
        return out.tolist()

    def push_to_pinecone(self, uploaded_file, index_name, pinecone_api_key, pinecone_env):
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=384,
                metric='cosine'
            )
            # wait for index to finish initialization
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)

        index = pinecone.Index(index_name)
        id = index.describe_index_stats()['total_vector_count']
        for file in uploaded_file:
            # what is the current index
            df_knowledge = pd.read_csv(file.file, header=None)
            df_dict = df_knowledge.to_dict(orient='index')
            for key in df_dict:
                metadata = [{"text": str(df_dict[key])}]
                embeddings = [self.embed_docs(str(df_dict[key]))]
                records = zip([str(id)], embeddings, metadata)
                index.upsert(vectors=records)
                id+=1
        time.sleep(3)


    def construct_context(self, contexts) -> str:
        max_section_len = 2000
        chosen_sections = []
        chosen_sections_len = 0

        for text in contexts:
            text = text.strip()
            # Add contexts until we run out of space.
            chosen_sections_len += len(text)
            if chosen_sections_len > max_section_len:
                break
            chosen_sections.append(text)
        concatenated_doc = ".".join(chosen_sections)
        return concatenated_doc

    def prepare_prompt(self, prompt, index_name):
        if index_name not in pinecone.list_indexes():
            return prompt, None

        prompt_template = """Answer the following QUESTION based on the CONTEXT
        given. If you do not know the answer and the CONTEXT doesn't
        contain the answer truthfully say "I don't know".

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        index = pinecone.Index(index_name)
        query_vec = self.embed_docs(prompt)
        res = index.query(query_vec, top_k=5, include_metadata=True)
        contexts = [match.metadata['text'] for match in res.matches]
        context_str = self.construct_context(contexts=contexts)
        prompt = prompt_template.replace("{context}", context_str).replace("{question}", prompt)
        return prompt, contexts
