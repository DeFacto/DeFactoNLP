class Evidence:

    def __init__(self, document="", sentence="", line_num=0):
        self.documents = set()
        self.sentences = set()
        self.pairs = set()
        # if arguments are passed
        if document != "":
            self.documents.add(document)
        if sentence != "":
            self.sentences.add(sentence)
        if line_num != 0:
            self.pairs.add((document, line_num))

    def add_document(self, doc):
        self.documents.add(doc)

    def add_sentence(self, sentence):
        self.sentences.add(sentence)

    def add_pair(self, doc, line_num):
        self.pairs.add((doc, line_num))
        self.add_document(doc)

    def get_difficulty_documents(self):
        return len(self.documents)

    def get_difficulty_sentences(self):
        return len(self.sentences)

    def get_difficulty(self):
        return len(self.pairs)
