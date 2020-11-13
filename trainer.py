import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from model import NumberPretrainerConfig, NumberPretrainer, Aggregator
import neptune

from utils import compute_metrics, get_label, MODEL_CLASSES

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, vocab_size=8002):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        self.config_numpre = NumberPretrainerConfig(output_size=self.num_labels)
        self.numpre_model = NumberPretrainer(self.config_numpre)
        self.aggregator = Aggregator(self.config_numpre)

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        # TODO: Revise the `vocab_size` after adding the shop_no
        self.config = self.config_class.from_pretrained(args.model_name_or_path,
                                                        num_labels=self.num_labels, 
                                                        finetuning_task=args.task,
                                                        output_hidden_states=True, output_attentions=True)
        print("Config loading complete!")
        self.bert = self.model_class(self.config)
        print("Current model: [{}]".format(self.bert))

        # GPU or CPU
        self.device = "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        print("Running on : [{}]".format(self.device))
        self.bert.to(self.device)
        self.numpre_model.to(self.device)
        print("Pre-Trained model loading complete!")

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        self.bert.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            tr_loss = 0.0
            tr_numpre_loss = 0.0
            tr_combined_loss = 0.0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.bert.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                numpre_inputs = {'input_val': batch[1], 'labels': batch[4]}
                bert_inputs = {'input_ids': batch[0],
                               'attention_mask': batch[2],
                               'labels': batch[4]}
                if self.args.model_type != 'distilkobert':
                    bert_inputs['token_type_ids'] = batch[3]
                outputs = self.bert(**bert_inputs)
                loss = outputs[0]  # `loss` BertForSequenceClassification
                bert_logits = outputs[1]  # `logits` from BertForSequenceClassification
                bert_hidden_states = outputs[2][0]
                numpre_out, numpre_loss = self.numpre_model(**numpre_inputs)  # NumberPretrainer loss
                aggregator_input = torch.cat((bert_hidden_states, numpre_out), 1)
                # print("bert_rep (shape): ", bert_rep.shape)
                # print("numpre_out (shape): ", numpre_out.shape)
                # print("aggregator_input (shape): ", aggregator_input.shape)
                
                combined_loss = numpre_loss + loss  # Multi-task loss

                if self.args.logger:
                    neptune.log_metric('BERT Loss', loss.item())
                    neptune.log_metric('NumPre Loss', numpre_loss.item())
                    neptune.log_metric('Combined Loss', combined_loss.item())

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                tr_numpre_loss += numpre_loss.item()
                tr_combined_loss += combined_loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.bert.parameters(), self.args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.numpre_model.parameters(), 0.5)  # TODO: Find appropriate args.max_grad_norm for self.numpre_model (=0.5 currently)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.bert.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()
                        logger.info("  Model saved at {}".format(self.args.model_dir))

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        
            if self.args.logger:
                neptune.log_metric('Training loss (per epoch)', tr_loss / len(self.train_dataset))
                neptune.log_metric('Training Numpre loss (per epoch)', tr_numpre_loss / len(self.train_dataset))
                neptune.log_metric('Training Combined loss (per epoch)', tr_combined_loss / len(self.train_dataset))

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        logger.info("  Dataset (dev) size = %d", len(dataset))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.bert.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.bert(**inputs)   # (loss), logits, (hidden_states), (attentions)
                tmp_eval_loss, logits = outputs[:2]
                print("Hidden states (shape)", outputs)
                exit()

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.bert.module if hasattr(self.bert, 'module') else self.bert
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert = self.model_class.from_pretrained(self.args.model_dir)
            self.bert.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
