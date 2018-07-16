import os

from copy import deepcopy
from typing import List, Union, Dict, Any

from overrides import overrides

from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary, Dataset, DataIterator, DatasetReader, Tokenizer, TokenIndexer, Token
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.models import Model, archive_model
from allennlp.training import Trainer
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB

import argparse
import logging
import sys
import json

from retrieval.reader import FEVERSentenceReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@WordSplitter.register('indexed_spaces')
class JustSpacesWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.  We use a somewhat odd name here to avoid coming too close to the more
    commonly used ``SpacyWordSplitter``.

    Note that we use ``sentence.split()``, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.
    """
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        tokens = [Token(text=t,idx=0) for t in sentence.split()]
        for id,token in enumerate(tokens):
            if id == 0:
                continue
            token.idx = tokens[id-1].idx + len(tokens[id-1].text) + 1
        return tokens
    @classmethod
    def from_params(cls, params: Params) -> 'WordSplitter':
        params.assert_empty(cls.__name__)
        return cls()

def train_model(db: FeverDocDB, params: Union[Params, Dict[str, Any]], cuda_device: int, serialization_dir: str) -> Model:
    """
    This function can be used as an entry point to running models in AllenNLP
    directly from a JSON specification using a :class:`Driver`. Note that if
    you care about reproducibility, you should avoid running code using Pytorch
    or numpy which affect the reproducibility of your experiment before you
    import and use this function, these libraries rely on random seeds which
    can be set in this function via a JSON specification file. Note that this
    function performs training and will also evaluate the trained model on
    development and test sets if provided in the parameter json.

    Parameters
    ----------
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    prepare_environment(params)

    os.makedirs(serialization_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), sys.stdout)  # type: ignore
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), sys.stderr)  # type: ignore
    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)

    with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    # Now we begin assembling the required parts for the Trainer.
    ds_params = params.pop('dataset_reader', {})
    dataset_reader = FEVERSentenceReader(db,
                                 wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                                 claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                                 token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})))

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    all_datasets: List[Dataset] = [train_data]
    datasets_in_vocab = ["train"]

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)
        all_datasets.append(validation_data)
        datasets_in_vocab.append("validation")
    else:
        validation_data = None

    logger.info("Creating a vocabulary using %s data.", ", ".join(datasets_in_vocab))
    vocab = Vocabulary.from_params(params.pop("vocabulary", {}),
                                   Dataset([instance for dataset in all_datasets
                                            for instance in dataset.instances]))
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    model = Model.from_params(vocab, params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))

    train_data.index_instances(vocab)
    if validation_data:
        validation_data.index_instances(vocab)

    trainer_params = params.pop("trainer")
    if cuda_device is not None:
        trainer_params["cuda_device"] = cuda_device
    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  validation_data,
                                  trainer_params)

    trainer.train()

    # Now tar up results
    archive_model(serialization_dir)

    return model



if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('param_path',
                           type=str,
                           help='path to parameter file describing the model to be trained')

    parser.add_argument("logdir",type=str)

    parser.add_argument("--cuda-device", type=int, default=None, help='id of GPU to use (if any)')


    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')


    args = parser.parse_args()

    db = FeverDocDB(args.db)
    params = Params.from_file(args.param_path)
    train_model(db,params,args.cuda_device, args.logdir)