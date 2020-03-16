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
            e = Evidence(str(pair[0]), str(pair[1]))
            self.predicted_evidence.append(e)

    def get_gold_documents(self):
        docs = set()
        for e in self.gold_evidence:
            docs |= e.documents
        return docs

    def calculate_corrected_docs(self, difficulty="all"):
        num_corr_docs = 0
        num_incorr_docs = 0
        gold_docs = self.get_gold_documents()
        if difficulty == "all":
            for doc in self.predicted_docs:
                if doc in gold_docs:
                    num_corr_docs += 1
                else:
                    num_incorr_docs += 1
        return num_corr_docs, num_incorr_docs

    @classmethod
    def find_by_id(cls, _id):
        return Claim.id_index[_id]

    @classmethod
    def document_retrieval_stats(cls, claims):
        precision_correct = 0
        recall_correct = 0
        total_claims = 0

        for claim in claims:
            if not claim.verifiable:
                continue
            total_claims += 1
            doc_correct, doc_incorrect = claim.calculate_corrected_docs(difficulty="all")

            precision_correct += doc_correct / (len(claim.predicted_docs) + 0.000001)
            recall_correct += doc_correct / (len(claim.get_gold_documents()) + 0.000001)

        precision_correct /= total_claims
        recall_correct /= total_claims

        return precision_correct, recall_correct
