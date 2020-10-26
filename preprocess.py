import pandas as pd
import csv
import os
from tqdm import tqdm
import logging
import re
import torch
from torch.utils.data import TensorDataset
from collections import defaultdict

logger = logging.getLogger(__name__)


class OrderExample(object):
    # A class for a data instance for Baemin Order dataset
    def __init__(
        self,
        mem_no=None,
        dvc_id=None,
        shop_no=None,  # The prime input
        shop_owner_no=None,
        rgn1_cd=None,
        rgn2_cd=None,
        rgn3_cd=None,
        ord_msg=None,
        ord_tm=None,
        item_name=None,
        review_yn=None,
        review_created_tm=None,
        rating=None,
        ord_price=None,
        delivery_yn=None,
        item_quantity=None,
        cpn_use_cnt=None,
        ord_dt=None,
        abuse_yn=None,
        abuse_type=None,
    ):
        self.mem_no = mem_no
        self.dvc_id = dvc_id
        self.shop_no = shop_no
        self.shop_owner_no = shop_owner_no
        self.rgn1_cd = rgn1_cd
        self.rgn2_cd = rgn2_cd
        self.rgn3_cd = rgn3_cd
        self.ord_msg = ord_msg
        self.ord_tm = ord_tm
        self.item_name = item_name
        self.review_yn = review_yn
        self.review_created_tm = review_created_tm
        self.rating = rating
        self.ord_price = ord_price
        self.delivery_yn = delivery_yn
        self.item_quantity = item_quantity
        self.cpn_use_cnt = cpn_use_cnt
        self.ord_dt = ord_dt
        self.abuse_yn = abuse_yn
        self.abuse_type = abuse_type

    def __repr__(self):
        sample_str = "\nShop_no: {}\nOrder_dt: {}\n".format(self.shop_no, self.ord_dt)
        return sample_str
    
    def __str__(self):
        sample_str = "\nShop_no: {}\nOrder_dt: {}\n".format(self.shop_no, self.ord_dt)
        return sample_str


class OrderFeatures(object):
    """A single data sample features"""
    def __init__(self, shop_no, input_ids, nontext_features, attention_mask, token_type_ids, label_id):
        self.shop_no = shop_no  # will serve as an `id` for per-shop_no embedding
        self.input_ids = input_ids
        self.nontext_features = nontext_features
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return "Shop_number: " + self.shop_no + "\n"

    # def to_dict(self):
    #     """Serializes this instance to a Python dictionary."""
    #     output = copy.deepcopy(self.__dict__)
    #     return output

    # def to_json_string(self):
    #     """Serializes this instance to a JSON string."""
    #     return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BaeminProcessor(object):
    """Processor for the woowahan data set """
    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        with open(input_file, "r") as fp:
            data_reader = csv.reader(fp)
            headers = next(data_reader)
            print(headers)
            df = pd.DataFrame(columns=headers)
            for i, row in enumerate(tqdm(data_reader)):
                df.loc[i] = row[:24]
                if i > 1000:  # TODO: set to 1000 instances due to speed limit
                    break
        print("Data Length: ", df.shape[0])
        print(df.iloc[0])
        return df

    def _create_examples(self, input_data, mode):
        examples = []
        vocab_shop_no = defaultdict(list)
        if mode == "train":
            num_data = input_data.shape[0]
            # Extract a single instance
            for i in tqdm(range(num_data), desc="(Train) Create Baemin Order Samples"):
                # Remove cancelled orders, no review instances (any other unnecessary attributes if any)
                if input_data.iloc[i]['review_yn'] != '0' or input_data.iloc[i]['ord_prog_cd'] != "주문취소":
                    # print(input_data.iloc[i])
                    ord_instance = input_data.iloc[i]
                    example = OrderExample(
                        mem_no=ord_instance['mem_no'],
                        dvc_id=ord_instance['dvc_id'],
                        shop_no=ord_instance['shop_no'],  # prime input
                        shop_owner_no=ord_instance['shop_owner_no'],
                        rgn1_cd=ord_instance['rgn1_cd'],
                        rgn2_cd=ord_instance['rgn2_cd'],
                        rgn3_cd=ord_instance['rgn3_cd'],
                        ord_msg=ord_instance['ord_msg'],
                        ord_tm=ord_instance['ord_tm'],
                        item_name=ord_instance['item_name'],
                        review_yn=ord_instance['review_yn'],
                        review_created_tm=ord_instance['review_created_tm'],
                        rating=ord_instance['rating'],
                        ord_price=ord_instance['ord_price'],
                        delivery_yn=ord_instance['delivery_yn'],
                        item_quantity=ord_instance['item_quantity'],
                        cpn_use_cnt=ord_instance['cpn_use_cnt'],
                        ord_dt=ord_instance['ord_dt']
                    )
                    examples.append(example)
                    # if ord_instance['shop_no'] not in vocab_shop_no:
                    vocab_shop_no[ord_instance['shop_no']] = None  # Creating a vocab_shop_no
            print("[Data Example List Created.]\n")
            print('Dataset length: ', len(examples))
            print("Unique shop_no: ", len(vocab_shop_no))

        elif mode == "dev" or mode == "test":
            num_data = input_data.shape[0]
            for i in tqdm(range(num_data), desc="(Dev) Create Baemin Order Samples"):
                ord_instance = input_data.iloc[i]
                example = OrderExample(
                    shop_no=ord_instance['shop_no'],  # prime input
                    ord_dt=ord_instance['ord_dt'],
                    abuse_yn=ord_instance['abuse_yn'],
                    abuse_type=ord_instance['abuse_type']
                )
                examples.append(example)
                vocab_shop_no[ord_instance['shop_no']] = None
                # TODO: Implement for validation.csv

        return examples, vocab_shop_no

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_filepath
        elif mode == 'dev':
            file_to_read = self.args.dev_filepath
        elif mode == 'test':
            file_to_read = self.args.test_filepath

        logger.info("DATA_PATH {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {
    "baemin": BaeminProcessor,
}


def create_vocab(examples):
    # TODO: Create vocabs from `shop_no`
    pass

# Reference: https://github.com/monologg/KoBERT-nsmc/blob/9c6f417748d82d7064b097e90030b1b68f351d9a/data_loader.py#L115
def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 mode="train"):
    if mode == "train":
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id

        features = []
        for (ex_index, example) in tqdm(enumerate(examples)):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            tokens = tokenizer.tokenize(' '.join([example.ord_msg, example.item_name])) # TODO: tokenize textual data

            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[:(max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            
            # Add [CLS] token
            tokens = [cls_token] + tokens
            cls_token_segment_id = 0
            token_type_ids = [cls_token_segment_id] + token_type_ids
            # print("Culprit! >> ", cls_token_segment_id) # TODO: Error: why is cls_token_segment_id == "train"?

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

            # Non-text input (item_quantity, cpn_use_cnt, ord_price, rating)
            # TODO: These features need normalization (except for binary features)
            if example.item_quantity.isnumeric() and example.cpn_use_cnt.isnumeric() and example.ord_price.isnumeric():
                item_quantity = int(example.item_quantity)
                cpn_use_cnt = int(example.cpn_use_cnt)
                ord_price = int(example.ord_price)
                rating = int(float(example.rating)) if len(example.rating) < 3 else 0
                nontext_features = [item_quantity, cpn_use_cnt, ord_price, rating]
            else:
                continue  # TODO: Case for erroneous instances

            '''
            1) Everything else is label `0`
            2) If len(item_name_list) < 2 and ord_price < 10000, then label `1` (order_fraud)
            3) If review_yn == 1 and (len(ord_msg) < 3 or ord_msg == null), then label `2` (review_fraud)
            4) If cpn_use_cnt == 0 and ord_price < 10000, then label `3` (coupon fraud)
            '''
            label_id = 0
            item_name_list = re.split(',|\+', example.item_name)
            if len(item_name_list) < 2 and int(example.ord_price) < 10000:
                label_id = 1
            elif int(example.review_yn) == 1 and (len(example.ord_msg) < 3 or example.ord_msg == 'null'):
                label_id = 2
            elif int(example.cpn_use_cnt) == 0 and int(example.ord_price) < 10000:
                label_id = 3

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("shop_no: %s" % example.shop_no)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s" % (label_id))

            features.append(
                OrderFeatures(shop_no=example.shop_no,
                              input_ids=input_ids,
                              nontext_features=nontext_features,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label_id=label_id
                            ))

    elif mode == "dev" or "test":
        pass

    return features


# TODO: Store the CLS representation produced by BERT as a representation for example.shop_no
def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len, mode)

    cached_vocabs_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), "vocabs", mode)

    cached_features_file = os.path.join(args.data_dir, cached_file_name)
    cached_vocabs_file = os.path.join(args.data_dir, cached_vocabs_name)
    if os.path.exists(cached_features_file) and os.path.exists(cached_vocabs_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        vocab_shop_no = torch.load(cached_vocabs_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples, vocab_shop_no = processor.get_examples("train")
        elif mode == "dev":
            examples, vocab_shop_no = processor.get_examples("dev")
        elif mode == "test":
            examples, vocab_shop_no = processor.get_examples("test")
        else:
            raise Exception("ModeError: Only train, dev and test modes are available")

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, mode)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_nontext_features = torch.tensor([f.nontext_features for f in features], dtype=torch.float)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    print('all_input_ids: ', all_input_ids.shape)
    print('all_label_ids: ', all_label_ids.shape)

    dataset = TensorDataset(all_input_ids, all_nontext_features, all_attention_mask,
                            all_token_type_ids, all_label_ids)

    print("Dataset Preprocessing Complete!")
    return dataset, vocab_shop_no
