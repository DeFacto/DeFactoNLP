from metrics.evidence import Evidence
from collections import defaultdict


class Claim:
    id_index = defaultdict(list)

    def __init__(self, _id, name, verifiable):
        self.id = _id
        self.name = name
        if verifiable == "VERIFIABLE":
            self.verifiable = 1
        else:
            self.verifiable = 0
        self.gold_evidence = []
        Claim.id_index[_id].append(self)
        self.predicted_docs = []
        self.predicted_evidence = []
        self.predicted_docs_ner = []
        self.predicted_evidence_ner = []

    def add_gold_evidence(self, document, evidence, line_num):
        evidence = Evidence(document, evidence, line_num)
        self.gold_evidence.append(evidence)

    def add_gold_evidences(self, evidences):
        for evidence in evidences:
            _evidence = Evidence()
            if len(evidence) > 1:  # needs more than 1 doc to be verifiable
                for e in evidence:
                    _evidence.add_pair(str(e[2]), str(e[3]))
            else:
                _evidence.add_pair(str(evidence[0][2]), str(evidence[0][3]))
            self.gold_evidence.append(_evidence)

    def add_predicted_docs(self, docs):
        for doc in docs:
            self.predicted_docs.append(doc)

    def add_predicted_sentences(self, pairs):
        for pair in pairs:
            e = str(pair[0]), str(pair[1])
            self.predicted_evidence.append(e)

    def add_predicted_docs_ner(self, docs):
        for doc in docs:
            self.predicted_docs_ner.append(doc)

    def add_predicted_sentences_ner(self, pairs):
        for pair in pairs:
            e = str(pair[0]), str(pair[1])
            self.predicted_evidence_ner.append(e)

    def get_gold_documents(self):
        docs = set()
        for e in self.gold_evidence:
            docs |= e.documents
        return docs

    def get_gold_pairs(self):
        pairs = set()
        for e in self.gold_evidence:
            pairs |= e.pairs
        return pairs

    def get_predicted_documents(self, _type="tfidf"):
        if _type == "tfidf":
            return self.predicted_docs
        if _type == "ner":
            return self.predicted_docs_ner
        else:
            documents = set()
            for doc in self.predicted_docs:
                documents.add(doc)
            for doc in self.predicted_docs_ner:
                documents.add(doc)
            return documents

    def get_predicted_evidence(self, _type="tfidf"):
        if _type == "tfidf":
            return self.predicted_evidence
        elif _type == "ner":
            return self.predicted_evidence_ner
        else:
            evidences = set()
            for e in self.predicted_evidence:
                evidences.add(e)
            for e in self.predicted_evidence_ner:
                evidences.add(e)
            return evidences

    def calculate_correct_docs(self, difficulty="all", _type="tfidf"):
        num_corr_docs = 0
        num_incorr_docs = 0
        gold_docs = self.get_gold_documents()
        if difficulty == "all":
            for doc in self.get_predicted_documents(_type=_type):
                if doc in gold_docs:
                    num_corr_docs += 1
                else:
                    num_incorr_docs += 1
        return num_corr_docs, num_incorr_docs

    def calculate_correct_sentences(self, difficulty="all", _type="tfidf"):
        num_corr_e = 0
        gold_pairs = self.get_gold_pairs()
        if difficulty == "all":
            for e in self.get_predicted_evidence(_type=_type):
                if e in gold_pairs:
                    num_corr_e += 1
        return num_corr_e

    def check_evidence_found_doc(self, _type="tfidf"):
        gold_docs = self.get_gold_documents()
        if _type == "tfidf":
            for doc in self.predicted_docs:
                if doc in gold_docs:
                    return True
            return False
        elif _type == "ner":
            for doc in self.predicted_docs_ner:
                if doc in gold_docs:
                    return True
            return False
        else:
            for doc in self.predicted_docs:
                if doc in gold_docs:
                    return True
            for doc in self.predicted_docs_ner:
                if doc in gold_docs:
                    return True
            return False

    @classmethod
    def find_by_id(cls, _id):
        return Claim.id_index[_id]

    @classmethod
    def document_retrieval_stats(cls, claims, _type="tfidf"):
        precision_correct = 0
        recall_correct = 0
        total_claims = 0

        for claim in claims:
            if not claim.verifiable:
                continue
            total_claims += 1
            doc_correct, doc_incorrect = claim.calculate_correct_docs(difficulty="all", _type=_type)

            precision_correct += doc_correct / (len(claim.get_predicted_documents(_type=_type)) + 0.000001)
            recall_correct += doc_correct / (len(claim.get_gold_documents()) + 0.000001)

        precision_correct /= total_claims
        recall_correct /= total_claims

        return precision_correct, recall_correct

    @classmethod
    def evidence_extraction_stats(cls, claims, _type="tfidf"):
        precision_sent_correct = 0
        recall_sent_correct = 0
        total_claims = 0

        precision_doc_sent_correct = 0
        recall_doc_sent_correct = 0
        total_claims_doc_found = 0

        for claim in claims:
            if not claim.verifiable:
                continue

            total_claims += 1
            sent_correct = claim.calculate_correct_sentences(difficulty="all", _type=_type)

            precision_sent_correct += sent_correct / (len(claim.get_predicted_evidence(_type=_type)) + 0.000001)
            recall_sent_correct += sent_correct / (len(claim.get_gold_pairs()) + 0.000001)

            if claim.check_evidence_found_doc(_type=_type):
                precision_doc_sent_correct += sent_correct / (len(claim.get_predicted_evidence(_type=_type)) + 0.000001)
                recall_doc_sent_correct += sent_correct / (len(claim.get_gold_pairs()) + 0.000001)
                total_claims_doc_found += 1

        precision_sent_correct /= total_claims
        recall_sent_correct /= total_claims

        precision_doc_sent_correct /= total_claims_doc_found
        recall_doc_sent_correct /= total_claims_doc_found

        return precision_sent_correct, recall_sent_correct, precision_doc_sent_correct, recall_doc_sent_correct
