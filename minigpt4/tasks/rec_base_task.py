"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import time

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

from minigpt4.common.dist_utils import is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.tasks.base_task import BaseTask


def uAUC_me(user, predict, label):
    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)  # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]

        k += 1
    print("only one interaction users:", only_one_interaction)
    auc = []
    only_one_class = 0

    for ui, pre_and_true in candidates_dict.items():
        pre_i, label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i, pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")

    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time() - start_time, 'uauc:', uauc)
    return uauc, computed_u, auc_for_user


# Function to gather tensors across processes
def gather_tensor(tensor, dst=0):
    if dist.is_available():
        world_size = dist.get_world_size()
        if world_size > 1:
            if not isinstance(tensor, list):
                tensor = [tensor]

            gathered_tensors = [torch.empty_like(t) for t in tensor]
            dist.gather(tensor, gathered_tensors, dst=dst)

            return gathered_tensors
        else:
            return tensor
    else:
        return tensor


class RecBaseTask(BaseTask):
    def valid_step(self, model, samples):
        """Forward pass to get regression predictions"""
        outputs = model(samples)
        return outputs

    def before_evaluation(self, model, dataset, **kwargs):
        pass
        # model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        """
        Process validation results and return dict with metrics for SwanLab logging
        
        Args:
            val_result: Result dict returned by evaluation method
            split_name: Dataset split name (e.g., 'valid', 'test')
            epoch: Current epoch
            
        Returns:
            dict: Dict containing validation metrics for SwanLab logging
        """
        if val_result is None:
            return {}
            
        # Ensure return format is dict
        if isinstance(val_result, dict):
            result_dict = val_result.copy()
        else:
            result_dict = {'agg_metrics': val_result}
        
        # Add epoch info
        result_dict['epoch'] = epoch
        
        print(f"Validation results {split_name} - epoch {epoch}: {result_dict}")
        
        # Log to SwanLab (backup logging in case runner logging fails)
        try:
            import swanlab
            log_dict = {}
            for k, v in result_dict.items():
                if isinstance(v, (int, float)):
                    log_dict[f"{split_name}/{k}"] = float(v)
                    
            if log_dict:
                swanlab.log(log_dict)
                print(f"SwanLab after_evaluation logged successfully: {list(log_dict.keys())}")
        except Exception as e:
            print(f"Warning: SwanLab after_evaluation logging failed: {e}")
        
        return result_dict

    def inference_step(self):
        raise NotImplementedError

    # def evaluation(self, model, data_loaders, cuda_enabled=True):
    #     model = model.eval()
    #     metric_logger = MetricLogger(delimiter="  ")
    #     auc_logger = MetricLogger(delimiter="  ")
    #     metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     auc_logger.add_meter("auc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    #     header = "Evaluation"
    #     # TODO make it configurable
    #     print_freq = len(data_loaders.loaders[0])//5 #10

    #     results = []
    #     results_loss = []
    #     results_logits = []
    #     labels = []
    #     k = 0
    #     use_auc = False
    #     for data_loader in data_loaders.loaders:
    #         for samples in metric_logger.log_every(data_loader, print_freq, header):
    #             # samples = next(data_loader)
    #             samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
    #             eval_output = self.valid_step(model=model, samples=samples)
    #             # results_loss.append(eval_output['loss'].item())
    #             if 'logits' in eval_output.keys():
    #                 use_auc = True
    #                 results_logits.extend(eval_output['logits'].detach().cpu().numpy())
    #                 labels.extend(samples['label'].detach().cpu().numpy())
    #                 logits = eval_output['logits']
    #                 logits[logits==0.5] = 1
    #                 acc = (logits-samples['label'])
    #                 acc = (acc==0).sum()/acc.shape[0]
    #                 metric_logger.update(acc=acc.item())
    #             else:
    #                 metric_logger.update(acc=0)
    #             # acc = accuracy_score(samples['label'].cpu().numpy().astype(int), logits.astype(int))
    #             # results.extend(eval_output)
    #             metric_logger.update(loss=eval_output['loss'].item())
    #             torch.cuda.empty_cache()

    #         if use_auc:
    #             auc = roc_auc_score(labels, results_logits)
    #             auc_logger.update(auc=auc)

    #         if is_dist_avail_and_initialized():
    #             dist.barrier()

    #         metric_logger.synchronize_between_processes()
    #         auc_logger.synchronize_between_processes()
    #         auc = 0
    #         # print("Label type......",type(labels),labels)
    #         if use_auc:
    #             auc = roc_auc_score(labels, results_logits)
    #         logging.info("Averaged stats: " + str(metric_logger.global_avg()) + " auc: " + str(auc) + "  global"+ str(auc_logger.global_avg()))

    #         if use_auc:
    #             results = {
    #                 'agg_metrics':auc,
    #                 'acc': metric_logger.meters['acc'].global_avg,
    #                 'loss':  metric_logger.meters['loss'].global_avg
    #             }
    #         else: # only loss usable
    #             results = {
    #                 'agg_metrics': -metric_logger.meters['loss'].global_avg,
    #             }

    #     return results
    def evaluation(self, model, data_loaders, cuda_enabled=True, split_name="valid"):
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        # TODO make it configurable
        print_freq = len(data_loaders.loaders[0]) // 5  # 10

        results = []
        results_loss = []

        k = 0
        eval_preds = []
        eval_targets = []
        from pathlib import Path
        debug_root = Path(self.result_dir if hasattr(self, 'result_dir') else '.') / 'debug' / split_name
        debug_root.mkdir(parents=True, exist_ok=True)
        max_debug_samples = 5
        dumped = 0

        for data_loader in data_loaders.loaders:
            for idx, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
                # samples = next(data_loader)
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                eval_output = self.valid_step(model=model, samples=samples)
                # results_loss.append(eval_output['loss'].item())
                # Collect regression predictions
                try:
                    if 'pred_rating' in eval_output and 'target_rating' in eval_output:
                        preds = eval_output['pred_rating'].detach().cpu().flatten().tolist()
                        tgts = eval_output['target_rating'].detach().cpu().flatten().tolist()
                        eval_preds.extend(preds)
                        eval_targets.extend(tgts)
                        # Debug sampling output for a few samples
                        if dumped < max_debug_samples:
                            try:
                                gen_texts = model.generate_sequence(samples, return_text=True)
                            except Exception:
                                gen_texts = None
                            uid = samples.get('UserID', None)
                            iid = samples.get('TargetItemID', None)
                            uid = uid.detach().cpu().tolist() if uid is not None else None
                            iid = iid.detach().cpu().tolist() if iid is not None else None
                            record = {
                                'uid': uid,
                                'iid': iid,
                                'gt_rating': tgts[:len(preds)],
                                'pred_rating': preds,
                                'gen_text': gen_texts,
                            }
                            import json
                            with open(debug_root / f"sample_{idx}_{dumped}.json", 'w', encoding='utf-8') as f:
                                json.dump(record, f, ensure_ascii=False, indent=2)
                            dumped += 1
                except Exception:
                    pass
                # Record loss (compatible with total_loss)
                loss_val = None
                if isinstance(eval_output, dict):
                    loss_val = eval_output.get('loss', eval_output.get('total_loss', None))
                if loss_val is None:
                    loss_float = 0.0
                else:
                    try:
                        loss_float = float(loss_val.detach().cpu().item())
                    except Exception:
                        try:
                            loss_float = float(loss_val)
                        except Exception:
                            loss_float = 0.0
                metric_logger.update(loss=loss_float)
                # SwanLab per-iter eval logging with reduced frequency
                try:
                    import swanlab
                    if idx % max(1, print_freq) == 0:
                        swanlab.log({f"eval/{split_name}/loss": loss_float})
                except Exception:
                    pass
                # torch.cuda.empty_cache()
            # Synchronize and compute regression metrics
            metric_logger.synchronize_between_processes()
            try:
                import numpy as np
                if len(eval_preds) > 0 and len(eval_preds) == len(eval_targets):
                    preds = np.array(eval_preds, dtype=float)
                    targets = np.array(eval_targets, dtype=float)
                    mae = float(np.mean(np.abs(preds - targets)))
                    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
                else:
                    mae, rmse = 0.0, 0.0
            except Exception:
                mae, rmse = 0.0, 0.0

            logging.info("Averaged stats: " + str(metric_logger.global_avg()) + f" rmse: {rmse:.4f} mae: {mae:.4f}")

            results = {
                'epoch': 0,
                'agg_metrics': -rmse,  # Lower is better, use negative as aggregate metric
                'loss': metric_logger.meters['loss'].global_avg,
                'rmse': rmse,
                'mae': mae,
            }
            # SwanLab per-split summary
            try:
                import swanlab
                swanlab.log({
                    f"eval/{split_name}/agg_metrics": float(results.get('agg_metrics', 0)),
                    f"eval/{split_name}/loss": float(metric_logger.meters['loss'].global_avg),
                    f"eval/{split_name}/rmse": float(rmse),
                    f"eval/{split_name}/mae": float(mae),
                })
            except Exception:
                pass

        return results

# class RecBaseTask:
#     def __init__(self, **kwargs):
#         super().__init__()

#         self.inst_id_key = "instance_id"

#     @classmethod
#     def setup_task(cls, **kwargs):
#         return cls()

#     def build_model(self, cfg):
#         model_config = cfg.model_cfg

#         model_cls = registry.get_model_class(model_config.arch)
#         return model_cls.from_config(model_config)

#     def build_datasets(self, cfg):
#         """
#         Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
#         Download dataset and annotations automatically if not exist.

#         Args:
#             cfg (common.config.Config): _description_

#         Returns:
#             dict: Dictionary of torch.utils.data.Dataset objects by split.
#         """

#         datasets = dict()

#         datasets_config = cfg.datasets_cfg

#         assert len(datasets_config) > 0, "At least one dataset has to be specified."

#         for name in datasets_config:
#             dataset_config = datasets_config[name]

#             builder = registry.get_builder_class(name)(dataset_config)
#             dataset = builder.build_datasets()

#             dataset['train'].name = name
#             if 'sample_ratio' in dataset_config:
#                 dataset['train'].sample_ratio = dataset_config.sample_ratio

#             datasets[name] = dataset

#         return datasets

#     def train_step(self, model, samples):
#         loss = model(samples)["loss"]
#         return loss

#     def valid_step(self, model, samples):
#         outputs = model.generate(samples)
#         return outputs
#         # raise NotImplementedError

#     def before_evaluation(self, model, dataset, **kwargs):
#         model.before_evaluation(dataset=dataset, task_type=type(self))

#     def after_evaluation(self, **kwargs):
#         pass

#     def inference_step(self):
#         raise NotImplementedError

#     def evaluation(self, model, data_loader, cuda_enabled=True):
#         metric_logger = MetricLogger(delimiter="  ")
#         metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
#         metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
#         header = "Evaluation"
#         # TODO make it configurable
#         print_freq = 10

#         results = []
#         results_loss = []
#         results_logits = []
#         labels = []

#         for samples in metric_logger.log_every(data_loader, print_freq, header):

#             samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
#             eval_output = self.valid_step(model=model, samples=samples)
#             results_loss.extend(eval_output['loss'])
#             results_logits.append(eval_output['logits'])
#             labels.append(samples['label'])
#             logits = eval_output['logits'].detach().cpu().numpy()
#             logits[logits>=0.5] = 1
#             acc = accuracy_score(samples['label'].cpu().numpy(), logits.int())
#             # results.extend(eval_output)
#             metric_logger.update(loss=eval_output['loss'].item())
#             metric_logger.update(acc=acc)


#         if is_dist_avail_and_initialized():
#             dist.barrier()

#         metric_logger.synchronize_between_processes()
#         auc = roc_auc_score(torch.cat(labels).detach().cpu().numpy(),torch.cat(results_logits).detach().cpu().numpy())
#         logging.info("Averaged stats: " + str(metric_logger.global_avg()) + "auc: " + str(auc))
#         results = {
#             'loss': torch.cat(results_loss).mean().item(),
#             'auc': auc
#         }

#         return results

#     def train_epoch(
#         self,
#         epoch,
#         model,
#         data_loader,
#         optimizer,
#         lr_scheduler,
#         scaler=None,
#         cuda_enabled=False,
#         log_freq=50,
#         accum_grad_iters=1,
#     ):
#         return self._train_inner_loop(
#             epoch=epoch,
#             iters_per_epoch=lr_scheduler.iters_per_epoch,
#             model=model,
#             data_loader=data_loader,
#             optimizer=optimizer,
#             scaler=scaler,
#             lr_scheduler=lr_scheduler,
#             log_freq=log_freq,
#             cuda_enabled=cuda_enabled,
#             accum_grad_iters=accum_grad_iters,
#         )

#     def train_iters(
#         self,
#         epoch,
#         start_iters,
#         iters_per_inner_epoch,
#         model,
#         data_loader,
#         optimizer,
#         lr_scheduler,
#         scaler=None,
#         cuda_enabled=False,
#         log_freq=50,
#         accum_grad_iters=1,
#     ):
#         return self._train_inner_loop(
#             epoch=epoch,
#             start_iters=start_iters,
#             iters_per_epoch=iters_per_inner_epoch,
#             model=model,
#             data_loader=data_loader,
#             optimizer=optimizer,
#             scaler=scaler,
#             lr_scheduler=lr_scheduler,
#             log_freq=log_freq,
#             cuda_enabled=cuda_enabled,
#             accum_grad_iters=accum_grad_iters,
#         )

#     def _train_inner_loop(
#         self,
#         epoch,
#         iters_per_epoch,
#         model,
#         data_loader,
#         optimizer,
#         lr_scheduler,
#         scaler=None,
#         start_iters=None,
#         log_freq=50,
#         cuda_enabled=False,
#         accum_grad_iters=1,
#     ):
#         """
#         An inner training loop compatible with both epoch-based and iter-based training.

#         When using epoch-based, training stops after one epoch; when using iter-based,
#         training stops after #iters_per_epoch iterations.
#         """
#         use_amp = scaler is not None

#         if not hasattr(data_loader, "__next__"):
#             # convert to iterator if not already
#             data_loader = iter(data_loader)

#         metric_logger = MetricLogger(delimiter="  ")
#         metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
#         metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

#         # if iter-based runner, schedule lr based on inner epoch.
#         logging.info(
#             "Start training epoch {}, {} iters per inner epoch.".format(
#                 epoch, iters_per_epoch
#             )
#         )
#         header = "Train: data epoch: [{}]".format(epoch)
#         if start_iters is None:
#             # epoch-based runner
#             inner_epoch = epoch
#         else:
#             # In iter-based runner, we schedule the learning rate based on iterations.
#             inner_epoch = start_iters // iters_per_epoch
#             header = header + "; inner epoch [{}]".format(inner_epoch)

#         for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
#             # if using iter-based runner, we stop after iters_per_epoch iterations.
#             if i >= iters_per_epoch:
#                 break

#             samples = next(data_loader)

#             samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
#             samples.update(
#                 {
#                     "epoch": inner_epoch,
#                     "num_iters_per_epoch": iters_per_epoch,
#                     "iters": i,
#                 }
#             )

#             lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 loss = self.train_step(model=model, samples=samples)

#             # after_train_step()
#             if use_amp:
#                 scaler.scale(loss).backward()
#             else:
#                 loss.backward()

#             # update gradients every accum_grad_iters iterations
#             if (i + 1) % accum_grad_iters == 0:
#                 if use_amp:
#                     scaler.step(optimizer)
#                     scaler.update()                     
#                 else:    
#                     optimizer.step()
#                 optimizer.zero_grad()

#             metric_logger.update(loss=loss.item())
#             metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#         # after train_epoch()
#         # gather the stats from all processes
#         metric_logger.synchronize_between_processes()
#         logging.info("Averaged stats: " + str(metric_logger.global_avg()))
#         return {
#             k: "{:.3f}".format(meter.global_avg)
#             for k, meter in metric_logger.meters.items()
#         }

#     @staticmethod
#     def save_result(result, result_dir, filename, remove_duplicate=""):
#         import json

#         result_file = os.path.join(
#             result_dir, "%s_rank%d.json" % (filename, get_rank())
#         )
#         final_result_file = os.path.join(result_dir, "%s.json" % filename)

#         json.dump(result, open(result_file, "w"))

#         if is_dist_avail_and_initialized():
#             dist.barrier()

#         if is_main_process():
#             logging.warning("rank %d starts merging results." % get_rank())
#             # combine results from all processes
#             result = []

#             for rank in range(get_world_size()):
#                 result_file = os.path.join(
#                     result_dir, "%s_rank%d.json" % (filename, rank)
#                 )
#                 res = json.load(open(result_file, "r"))
#                 result += res

#             if remove_duplicate:
#                 result_new = []
#                 id_list = []
#                 for res in result:
#                     if res[remove_duplicate] not in id_list:
#                         id_list.append(res[remove_duplicate])
#                         result_new.append(res)
#                 result = result_new

#             json.dump(result, open(final_result_file, "w"))
#             print("result file saved to %s" % final_result_file)

#         return final_result_file
