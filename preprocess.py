import pandas as pd
import numpy as np
import csv
import os
from tqdm import tqdm
import logging
import re
import torch
from torch.utils.data import TensorDataset
# from collections import defaultdict
# import pickle

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
        abuse_class=None,
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
        self.abuse_class = abuse_class

    def __repr__(self):
        sample_str = "\nShop_no: {}\nOrder_dt: {}\n".format(self.shop_no, self.ord_dt)
        return sample_str
    
    def __str__(self):
        sample_str = "\nShop_no: {}\nOrder_dt: {}\n".format(self.shop_no, self.ord_dt)
        return sample_str


class OrderFeatures(object):
    """A single data sample features"""
    def __init__(self, input_ids, nontext_features=None, attention_mask=None, token_type_ids=None, label_id=None):
        # self.shop_no = shop_no  # will serve as an `id` for per-shop_no embedding
        self.input_ids = input_ids
        self.nontext_features = nontext_features
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return "Shop_number: " + self.shop_no + "\n"


class BaeminProcessor(object):
    """Processor for the woowahan data set """
    def __init__(self, args):
        self.args = args
        # `shop_no` in the validation file that have `abuse_types`
        self.shop_no_ex = ["0299E61C687A81580E217EC98EEA7743", "1ADEBC59634C2D13B170FB7E96D6C57F", "2F5014BCC29966F667AF555FA6FAB015", "2FF539B930397C6005F13A6AA6D31658", 
            "33115FB448E2D821BCA3C41BE01E294F", "3660CD934F4C1BB184F39768CF08180B", "3F7B70727D3AACAF116FB5A805FC952C", "4B98850839AB4FBD95C3CCE889D59E94",
            "5F039D1D6E7C3CF2E6786E3594DA8E9E", "604F7BC10755B4111D32CB2281873EB2", "6602304D32A9BBEA3C7E87C51DFF247A", "72660CFE9BF4AB00A4F01B1F5441B1AE",
            "7B721777CECE119F097488CCF2BE7475", "7B9F043578B1DC1B35704E0CA76A2E30", "9FC6DAEFF7BB74B577ABD0CFD9ADF426", "A2DE5086981D6BA2B20A33C25E3A85CA",
            "A2F966D6B4CA78689F95A6EC81F60F75", "B2066E681DC94C3C42AB2D83D2B5B0B4", "B9ACF3CBA5A0957F493D40BF169C3995", "BE7393A50ADE41B8E14B7AB50DA12C20",
            "CD266107DE0619329079F63C8C85F3CD", "CDA834B36D0FCFF646F9BAD9226824B2", "E19B247FB7FB5C08A0F6789FEE3A47A1", "F94405DD8B1206EB522569957FFCFA27"]

    # @classmethod
    def _read_file(self, input_file, quotechar=None):
        if self.args.train_filepath == "train.csv":
            # Handles erroneous row with more than 25 features
            with open(input_file, "r") as fp:
                data_reader = csv.reader(fp)
                headers = next(data_reader)
                if "shop_no" in headers[0]:
                    headers[0] = "shop_no"
                    print("Headers (dev) >> ", headers)
                else:
                    print("Headers (train) >> ", headers)

                df = pd.DataFrame(columns=headers)
                for i, row in enumerate(tqdm(data_reader)):
                    df.loc[i] = row[:24]
                    # if i > 1000:  # TODO: set to 1000 instances due to speed limit
                    #     break
        elif self.args.train_filepath == "abuse_train.csv" or self.args.dev_filepath == "abuse_dev.csv":
            df = pd.read_csv(input_file)
            df = df.dropna()
            df = df.drop(df.columns[:2], axis=1)  # Drop the `Unnamed` columns

        print("Data read complete (samples) >> ", df.iloc[:10])
        print("Data Length >> ", df.shape[0])
        return df

    def _create_examples(self, input_data, mode):
        examples = []
        vocab_shop_no = {}
        if mode == "train" or (self.args.dev_filepath == "abuse_dev.csv" and mode == "dev"):
            num_data = input_data.shape[0]
            # Extract a single instance
            for i in tqdm(range(num_data), desc="({}) Create Baemin Order Samples".format(mode)):
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
                    vocab_shop_no[example.shop_no] = None
            print("[Data Example List Created.]\n")
            print('Dataset length: ', len(examples))

        elif self.args.dev_filepath == "validation.csv" and (mode == "dev" or mode == "test"):
            num_data = input_data.shape[0]
            for i in tqdm(range(num_data), desc="({}) Create Baemin Order Samples".format(mode)):
                ord_instance = input_data.iloc[i]
                # print("ord_instance[shop_no]: ", ord_instance)
                example = OrderExample(
                    shop_no=ord_instance['shop_no'],  # prime input
                    ord_dt=ord_instance['ord_dt'],
                    abuse_yn=ord_instance['abuse_yn'],
                    abuse_class=ord_instance['abuse_class']
                )
                examples.append(example)

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


def convert_examples_to_features(examples, max_seq_len, tokenizer, file_path,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 mode=None):
    print("Current mode >> {}\n".format(mode))
    cls_token = tokenizer.cls_token  # cls_token_id == 2
    sep_token = tokenizer.sep_token  # sep_token_id == 3
    pad_token_id = tokenizer.pad_token_id  # pad_token_id == 1
    # print('cls_token >> ', cls_token)
    # print('sep_token >> ', sep_token)
    # print('pad_token_id >> ', pad_token_id)
    
    features = []
    if mode == "train" or (file_path == "abuse_dev.csv" and mode == "dev"):
        # Setting based on the current model type
        for (ex_index, example) in tqdm(enumerate(examples)):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
                logger.info("PATH %s" % (file_path))

            tokens = tokenizer.tokenize(' '.join([example.ord_msg, example.item_name]))

            # Add `shop_no` to vocabulary via `tokenizer.add_tokens("new_token" or list of tokens)`
            # tokenizer.add_tokens(example.shop_no)

            # Account for [CLS] and [SEP] - Trim the token sequence
            special_tokens_count = 3
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[:(max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            
            # Add [CLS] token
            tokens = [cls_token] + [example.shop_no[:10]] + [sep_token] + tokens
            token_type_ids += [sequence_a_segment_id] * 2

            cls_token_segment_id = 0
            token_type_ids = [cls_token_segment_id] + token_type_ids
            # print("Culprit! >> ", cls_token_segment_id) # TODO: Error: why is cls_token_segment_id == "train"?
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length) if padding_length >= 0 else input_ids[:max_seq_len]
            # input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length) if padding_length >= 0 else attention_mask[:max_seq_len]
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length) if padding_length >= 0 else token_type_ids[:max_seq_len]
            # print(tokenizer.convert_ids_to_tokens(input_ids))
            # print(input_ids)
            # print(len(input_ids))
            assert len(input_ids) == max_seq_len, "(Train) Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "(Train) Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "(Train) Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

            # Non-text input (item_quantity, cpn_use_cnt, ord_price, rating)
            # TODO: These features need normalization (except for binary features)
            if type(example.item_quantity) == str and type(example.cpn_use_cnt) == str and type(example.ord_price) == str:
                if example.item_quantity.isnumeric() and example.cpn_use_cnt.isnumeric() and example.ord_price.isnumeric():
                    item_quantity = int(example.item_quantity)
                    cpn_use_cnt = int(example.cpn_use_cnt)
                    ord_price = int(example.ord_price)
            elif type(example.item_quantity) == int and type(example.cpn_use_cnt) == int and type(example.ord_price) == int:
                item_quantity = example.item_quantity
                cpn_use_cnt = example.cpn_use_cnt
                ord_price = example.ord_price
            else:
                continue

            if type(example.rating) == str and not example.rating.isnumeric():
                rating = 0
            elif type(example.rating) == str and example.rating.isnumeric():
                rating = int(float(example.rating))
            else:
                rating = int(float(example.rating))
            nontext_features = [item_quantity, cpn_use_cnt, ord_price, rating]

            '''
            1) Everything else is label `0`
            2) If len(item_name_list) < 2 and ord_price < 10000, then label `1` (order_fraud)
            3) If review_yn == 1 and (len(ord_msg) < 3 or ord_msg == null), then label `2` (review_fraud)
            4) If cpn_use_cnt == 0 and ord_price < 10000, then label `3` (coupon fraud)
            '''
            # label_id = 0
            # item_name_list = re.split(',|\+', example.item_name)
            # if len(item_name_list) < 2 and int(example.ord_price) < 10000:
            #     label_id = 1
            # elif int(example.review_yn) == 1 and (len(example.ord_msg) < 3 or example.ord_msg == 'null'):
            #     label_id = 2
            # elif int(example.cpn_use_cnt) == 0 and int(example.ord_price) < 10000:
            #     label_id = 3

            '''
            delivery_yn == "배달" , then label_id = 0
            delivery_yn == "배민오더" , then label_id = 1
            '''
            label_id = 0
            if example.delivery_yn == "배달":
                label_id = 0
            elif example.delivery_yn == "배민오더":
                label_id = 1
            else:
                label_id = 1

            if ex_index < 10:
                logger.info("*** Example ***")
                logger.info("shop_no: %s" % example.shop_no)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s" % (label_id))

            features.append(
                OrderFeatures(input_ids=input_ids,
                              nontext_features=nontext_features,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label_id=label_id
                            ))

    elif mode == "dev" or "test":
        # shop_no, ord_dt, abuse_yn, abuse_class
        for (ex_index, example) in tqdm(enumerate(examples)):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            # Add `shop_no` to vocabulary via `tokenizer.add_tokens("new_token" or list of tokens)`
            tokens = tokenizer.tokenize(example.shop_no)

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
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_len, "(Dev) Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "(Dev) Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "(Dev) Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

            # label_id = 0
            # if example.abuse_yn != "0":
            #     if example.abuse_class == "1":
            #         label_id = 1
            #     elif example.abuse_class == "2":
            #         label_id = 2
            #     elif example.abuse_class == "3":
            #         label_id = 3
            #     else:
            #         raise Exception("Only labels 0, 1, 2 and 3 are available  >> Error label_id: {}".format(example.abuse_class))

            # New labels according to `devliery_yn` =>  `배달` or `포장`
            label_id = 0
            if example.delivery_yn == "배달":
                label_id = 0
            elif example.delivery_yn == "배민오더":
                label_id = 1
            else:
                label_id = 1

            if ex_index < 10:
                logger.info("*** Example ***")
                logger.info("shop_no: %s" % example.shop_no)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s" % (label_id))

            features.append(
                OrderFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label_id=label_id
                            )
            )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len, mode)

    cached_features_file = os.path.join(args.data_dir, cached_file_name)
    vocab_size = 8002
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples, vocab_shop_no = processor.get_examples("train")
        elif mode == "dev":
            examples, _ = processor.get_examples("dev")
        elif mode == "test":
            examples, _ = processor.get_examples("test")
        else:
            raise Exception("ModeError: Only train, dev and test modes are available")
        print("[ load_and_cache_examples: Mode >> {} ]".format(mode))

        file_path = args.dev_filepath if mode == "dev" else args.train_filepath

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, file_path=file_path, mode=mode)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    if mode == "train" or (args.dev_filepath == "abuse_dev.csv" and mode == "dev"):
        all_nontext_features = torch.tensor([f.nontext_features for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_nontext_features, all_attention_mask, all_token_type_ids, all_label_ids)
    elif mode == "dev" or mode == "test":
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)

    print("Dataset Preprocessing Complete!")
    return dataset, vocab_size
