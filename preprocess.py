import pandas as pd
import os
from tqdm import tqdm
from typing import Any, Dict


class OrderExample(object):
    # A class for a data instance for Baemin Order dataset
    def __init__(
        self,
        mem_no,
        dvc_id,
        shop_no,
        shop_owner_no,
        rgn1_cd,
        rgn2_cd,
        rgn3_cd,
        ord_msg,
        ord_tm,
        review_created_tm,
        rating,
        ord_price,
        delivery_yn,
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
        self.review_created_tm = review_created_tm
        self.rating = rating
        self.ord_price = ord_price
        self.delivery_yn = delivery_yn

    def __repr__(self):
        sample_str = "Member_no: {}\nOrder_msg: {}\nRating: {}\nDelivery: {}\n".format(self.mem_no, self.ord_msg, self.rating, self.delivery_yn)
        return sample_str
    
    def __str__(self):
        sample_str = "Member_no: {}\nOrder_msg: {}\nRating: {}\nDelivery: {}\n".format(self.mem_no, self.ord_msg, self.rating, self.delivery_yn)
        return sample_str


def _create_sample(input_data):
    examples = []
    # Extract a single instance
    for _, data in tqdm(input_data.iterrows(), desc="Create Baemin Order Samples"):
        # TODO: 주문취소, review 없는거 제거 (and other unnecessary data samples)
        example = OrderExample(
            mem_no=data['mem_no'],
            dvc_id=data['dvc_id'],
            shop_no=data['shop_no'],
            shop_owner_no=data['shop_owner_no'],
            rgn1_cd=data['rgn1_cd'],
            rgn2_cd=data['rgn2_cd'],
            rgn3_cd=data['rgn3_cd'],
            ord_msg=data['ord_msg'],
            ord_tm=data['ord_tm'],
            review_created_tm=data['review_created_tm'],
            rating=data['rating'],
            ord_price=data['ord_price'],
            delivery_yn=data['delivery_yn'],
        )
        examples.append(example)

    print('Dataset length: ', len(examples))
    return examples


def get_dataset(filepath, filename=None):
    order_df = pd.read_csv(filepath, encoding="cp949") # cp949 - windows encoding for Korean
    dataset = _create_sample(order_df)
    return dataset


# def hotpot_features(question, context, config=BertConfig, tokenizer=BertTokenizerFast):
#     encoded_dict = tokenizer.encode_plus(question, context, max_length=config.max_position_embeddings)
#     input_ids = encoded_dict["input_ids"]
#     token_type_ids = encoded_dict["token_type_ids"]
#     return input_ids, token_type_ids


class BaeminDataset(object):
    def __init__(self, args):
        if args.data_dir is not None:
            self.path = os.path.join(args.data_dir, args.filepath)
        else:
            self.path = args.filepath
        self.dataset = get_dataset(self.path)
        print("Data sample: ", self.dataset)

    def __len__(self) -> int:
        length = len(self.dataset)
        return length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.dataset[index],
        sample: Dict[str, Any] = {
            'mem_no': instance.ord_no,
            'dvc_id': instance.dvc_id,
            'shop_no': instance.shop_no,
            'shop_owner_no': instance.shop_owner_no,
            'rgn1_cd': instance.rgn1_cd,
            'rgn2_cd': instance.rgn2_cd,
            'rgn3_cd': instance.rgn3_cd,
            'ord_msg': instance.ord_msg,
            'ord_tm': instance.ord_tm,
            'review_created_tm': instance.review_created_tm,
            'rating': instance.rating,
            'ord_price': instance.ord_price,
            'delivery_yn': instance.delivery_yn,
        }

        return sample

def convert_examples_to_features():
    # Reference: https://github.com/monologg/KoBERT-nsmc/blob/9c6f417748d82d7064b097e90030b1b68f351d9a/data_loader.py#L115
    pass