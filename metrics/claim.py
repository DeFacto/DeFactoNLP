from metrics.evidence import Evidence


class Claim:

    def __init__(self, _id, name, verifiable):
        self.id = _id
        self.name = name
        if verifiable == "VERIFIABLE":
            self.verifiable = 1
        else:
            self.verifiable = 0
        self.gold_evidence = []

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

    def get_gold_documents(self):
        docs = set()
        for e in self.gold_evidence:
            docs |= e.documents
        return docs
