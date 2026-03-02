
import spacy

class SpacySplitter:
    r"""A utility class that uses spaCy to split definitions into smaller chunks. This is useful for creating more training samples from each definition, which can help improve the performance of machine learning models that learn associations between words and their definitions.
    """
    nlp: spacy.language.Language
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def split_definition(self, definition: str) -> list[str]:
        r"""Splits a definition into smaller chunks using spaCy's noun chunks and clause detection.
        This helps create more training samples from each definition, improving the model's ability to learn associations between words and their definitions.
        Args:
            definition (str): The definition text to split.
        Returns:
            list[str]: A list of smaller chunks extracted from the definition.
        """

        doc = self.nlp(definition)
    
        chunks = []
        for sent in doc.sents:
            chunks.extend([str(chunk).strip() for chunk in sent.noun_chunks])
            chunks.extend([str(clause).strip() for clause in sent.root.subtree])

        return list(set(chunks))