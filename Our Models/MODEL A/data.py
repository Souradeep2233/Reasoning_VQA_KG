import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import clip
import spacy
import gensim
import numpy as np

# --- 1. The ConceptNet Feature Extractor Class ---
class ConceptNetFeatureExtractor:
    """
    This class loads the ConceptNet embeddings and processes
    text to extract an aggregated feature vector.
    """
    def __init__(self, embedding_path):
        print(f"Loading spaCy model 'en_core_web_sm'...")
        self.nlp = spacy.load("en_core_web_sm")
        
        print(f"Loading ConceptNet embeddings from {embedding_path}...")
        print("This can take several minutes and use a lot of RAM.")
        self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            embedding_path
        )
        self.embed_dim = self.embeddings.vector_size
        print(f"Embeddings loaded. Vector size: {self.embed_dim}")

    def get_features(self, text):
        doc = self.nlp(text.lower())
        concepts = [
            token.lemma_ 
            for token in doc 
            if not token.is_stop and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
        ]
        
        vectors = []
        for concept in concepts:
            concept_uri = f"/c/en/{concept}"
            if concept_uri in self.embeddings:
                vectors.append(self.embeddings[concept_uri])
        
        if not vectors:
            agg_vector = np.zeros(self.embed_dim, dtype=np.float32)
        else:
            agg_vector = np.mean(vectors, axis=0).astype(np.float32)
            
        return torch.from_numpy(agg_vector)

# --- 2. The VQA Dataset Class ---
# (I fixed a typo in the class name here: RealVQADataset)
class RealVQADataset(Dataset):
    def __init__(self, clip_preprocess, bert_tokenizer, feature_extractor, split):
        self.clip_preprocess = clip_preprocess
        self.bert_tokenizer = bert_tokenizer
        self.feature_extractor = feature_extractor
        self.cn_embed_dim = self.feature_extractor.embed_dim
        
        print(f"Loading Hugging Face dataset 'lmms-lab/VQAv2' (split: {split})...")
        self.hf_dataset = load_dataset("lmms-lab/VQAv2", split=split)
        print(f"Split '{split}' loaded successfully with {len(self.hf_dataset)} samples.")

    def __len__(self):
        return len(self.hf_dataset)

    def _get_most_common_answer(self, answers_list):
        if not answers_list:
            return "[PAD]"
            
        if isinstance(answers_list[0], dict):
            answer_texts = [ans['answer'] for ans in answers_list]
        else:
            answer_texts = answers_list

        if not answer_texts:
            return "[PAD]"
            
        counter = Counter(answer_texts)
        return counter.most_common(1)[0][0]

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample['image'].convert("RGB")
        image_processed = self.clip_preprocess(image)
        
        question_text = sample['question']
        question_tokenized = clip.tokenize([question_text])[0]
        
        cn_features = self.feature_extractor.get_features(question_text)
        
        most_common_answer = self._get_most_common_answer(sample['answers'])
        
        label_id = self.bert_tokenizer.encode(
            most_common_answer, 
            add_special_tokens=False,
            max_length=1,
            truncation=True
        )
        
        if not label_id:
            label_id = [self.bert_tokenizer.pad_token_id]
            
        label = torch.tensor(label_id[0]).long()
        
        return image_processed, question_tokenized, cn_features, label

# --- 3. THE FIX IS HERE ---
# Notice the new 'split' and 'shuffle' arguments in the function definition.
def get_vqa_loader(batch_size, clip_preprocess, bert_tokenizer, feature_extractor, split, shuffle=True):
    """
    MODIFIED: Now accepts a 'split' string and 'shuffle' boolean
    to be more flexible.
    """
    dataset = RealVQADataset(
        clip_preprocess=clip_preprocess, 
        bert_tokenizer=bert_tokenizer, 
        feature_extractor=feature_extractor,
        split=split  # Pass the split string to the dataset
    )
    
    # Use a smaller batch size if the dataset split is smaller
    if len(dataset) < batch_size:
        print(f"Warning: Dataset split '{split}' size ({len(dataset)}) is smaller than batch size ({batch_size}).")
        print(f"Setting batch size for this loader to {len(dataset)}.")
        batch_size = len(dataset)
        if batch_size == 0:
            print(f"Error: Split '{split}' has 0 samples. Loader will be empty.")
            return None # Return None if dataset is empty

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, # Use the shuffle argument
        num_workers=0 
    )
    
    return loader