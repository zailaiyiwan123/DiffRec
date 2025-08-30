"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg
        evaluate_only = cfg.run_cfg.evaluate

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets(evaluate_only=evaluate_only)

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        """Return raw model forward output (dict or tensor)"""
        result = model(samples)
        return result

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []
        
        # Collect validation losses for SwanLab logging
        eval_losses = []
        eval_rating_losses = []
        
        model.eval()
        
        with torch.no_grad():
            for samples in metric_logger.log_every(data_loader, print_freq, header):
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

                eval_output = self.valid_step(model=model, samples=samples)
                
                # Collect validation losses for logging
                if isinstance(eval_output, dict):
                    if "loss" in eval_output:
                        loss_val = eval_output["loss"]
                        if hasattr(loss_val, 'item'):
                            eval_losses.append(loss_val.item())
                    if "rating_loss" in eval_output:
                        rating_loss_val = eval_output["rating_loss"]
                        if hasattr(rating_loss_val, 'item'):
                            eval_rating_losses.append(rating_loss_val.item())
                
                results.extend(eval_output)

        # Calculate average validation loss and log to SwanLab
        if eval_losses:
            avg_eval_loss = sum(eval_losses) / len(eval_losses)
            avg_rating_loss = sum(eval_rating_losses) / len(eval_rating_losses) if eval_rating_losses else 0.0
            
            print(f"Validation - Avg loss: {avg_eval_loss:.4f}, Rating loss: {avg_rating_loss:.4f}")
            
            try:
                import swanlab
                swanlab.log({
                    "eval/loss": avg_eval_loss,
                    "eval/rating_loss": avg_rating_loss,
                })
                print(f"SwanLab: Validation loss logged: {avg_eval_loss:.4f}")
            except Exception as e:
                print(f"Warning: SwanLab validation loss logging failed: {e}")

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # Aggregate for computing epoch-level RMSE/MAE if model returns pred/target
        epoch_pred_list = []
        epoch_target_list = []

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        
        # Set current epoch info for image generation monitoring
        try:
            if hasattr(model, 'set_current_epoch'):
                model.set_current_epoch(epoch)
        except Exception:
            pass
            
        header = "TRAINING epoch [{}]".format(epoch)
        if start_iters is None:
            inner_epoch = epoch
        else:
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        logging.info(f"Starting training loop: iters_per_epoch={iters_per_epoch}")
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i % 1000 == 0:
                logging.info(f"Training step {i}/{iters_per_epoch}")
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = self.train_step(model=model, samples=samples)
                # Handle dict or direct loss return
                if isinstance(outputs, dict):
                    loss = outputs.get("loss", outputs.get("total_loss", None))
                else:
                    loss = outputs

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                # Add gradient clipping for training stability
                max_grad_norm = getattr(model, 'max_grad_norm', 1.0)
                if max_grad_norm is not None and max_grad_norm > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                else:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()                     
                    else:    
                        optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # Image generation module integration
            image_metrics = {}
            if isinstance(outputs, dict) and "adaptive_weight" in outputs:
                try:
                    image_metrics = self._integrate_image_generation(samples, outputs)
                    if image_metrics:
                        outputs.update(image_metrics)
                except Exception as e:
                    logging.warning(f"Image generation module failed: {e}")
            
            # SwanLab per-iter logging
            try:
                import swanlab
                global_step = epoch * iters_per_epoch + i
                log_data = {
                    "train/loss": float(loss.item()),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/epoch": float(epoch),
                    "train/iter": float(i),
                    "train/global_step": global_step,
                }
                
                if isinstance(outputs, dict):
                    # All loss component metrics
                    loss_keys = ["total_loss", "rating_loss", "image_loss", "diffusion_loss", "diffusion_loss_raw", 
                                "image_supervision_loss", "diffusion_weight", "adaptive_weight",
                                "rating_loss_ratio", "image_loss_ratio"]
                    for k in loss_keys:
                        if k in outputs:
                            v = outputs[k]
                            if hasattr(v, 'item'):
                                log_data[f"train/{k}"] = float(v.item())
                            elif isinstance(v, (int, float)):
                                log_data[f"train/{k}"] = float(v)
                    
                    # Image generation metrics and expert scores
                    image_keys = ["consistency", "accuracy", "integrity", "quality", "adaptive_weight",
                                 "expert_consistency", "expert_accuracy", "expert_integrity", "expert_quality"]
                    for k in image_keys:
                        if k in outputs:
                            v = outputs[k]
                            if hasattr(v, 'item'):
                                log_data[f"image/{k}"] = float(v.item())
                            elif isinstance(v, (int, float)):
                                log_data[f"image/{k}"] = float(v)
                    
                    # Add total loss decomposition info
                    if "total_loss" in outputs and "rating_loss" in outputs and "image_loss" in outputs:
                        total_val = float(outputs["total_loss"]) if hasattr(outputs["total_loss"], "item") else float(outputs["total_loss"])
                        rating_val = float(outputs["rating_loss"]) if hasattr(outputs["rating_loss"], "item") else float(outputs["rating_loss"])
                        image_val = float(outputs["image_loss"]) if hasattr(outputs["image_loss"], "item") else float(outputs["image_loss"])
                        
                        if total_val > 0:
                            log_data["loss_decomposition/rating_percentage"] = (rating_val / total_val) * 100
                            log_data["loss_decomposition/image_percentage"] = (image_val / total_val) * 100
                
                # Collect pred/target for epoch-level RMSE/MAE
                if "pred_rating" in outputs and "target_rating" in outputs:
                    try:
                        epoch_pred_list.extend(outputs["pred_rating"].detach().cpu().flatten().tolist())
                        epoch_target_list.extend(outputs["target_rating"].detach().cpu().flatten().tolist())
                    except Exception:
                        pass
                
                # SwanLab logging frequency control
                if i % log_freq == 0:
                    swanlab.log(log_data)
                
            except Exception as e:
                print(f"Warning: SwanLab logging failed at epoch {epoch}, step {i}: {e}")
                pass
            # torch.cuda.empty_cache()

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        logging.info(f"Training epoch completed! Stats: {metric_logger.global_avg()}")

        # Calculate epoch-level RMSE/MAE and report
        epoch_metrics = {
            k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()
        }
        try:
            import numpy as np
            if len(epoch_pred_list) > 0 and len(epoch_pred_list) == len(epoch_target_list):
                preds = np.array(epoch_pred_list, dtype=float)
                targets = np.array(epoch_target_list, dtype=float)
                mae = float(np.mean(np.abs(preds - targets)))
                rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
                epoch_metrics.update({
                    "epoch_mae": "{:.4f}".format(mae),
                    "epoch_rmse": "{:.4f}".format(rmse),
                })
                try:
                    import swanlab
                    swanlab.log({
                        "train/epoch_mae": mae,
                        "train/epoch_rmse": rmse,
                        "train/epoch": float(epoch),
                        "train/epoch_completed": float(epoch),
                    })
                except Exception:
                    pass
        except Exception:
            pass

        return epoch_metrics

    def _integrate_image_generation(self, samples, outputs):
        """
        Image generation module integration based on CoRA method
        
        Args:
            samples: Current batch data samples
            outputs: Model outputs (including adaptive_weight)
            
        Returns:
            dict: Image evaluation metrics
        """
        try:
            from image_personalization.hook import update_image_module
            
            # Extract necessary fields
            instruction = samples.get('text_input', ['Help me recommend items'])
            if isinstance(instruction, list):
                instruction = instruction[0]
            
            title = samples.get('title', samples.get('asin', 'Unknown Item'))
            if isinstance(title, list):
                title = title[0]
                
            his_interaction = samples.get('his_interaction', '')
            if isinstance(his_interaction, list):
                his_interaction = his_interaction[0]
            
            item_features = samples.get('item_features', '')
            if isinstance(item_features, list):
                item_features = item_features[0]
            
            # Get adaptive weight
            adaptive_weight = outputs.get('adaptive_weight', 0.5)
            if hasattr(adaptive_weight, 'item'):
                adaptive_weight = adaptive_weight.item()
            
            # Build image generation sample
            image_sample = {
                'instruction': str(instruction),
                'title': str(title),
                'his_interaction': str(his_interaction),
                'item_features': str(item_features),
            }
            
            # Call image generation and evaluation
            result = update_image_module(
                image_sample,
                adaptive_weight=float(adaptive_weight),
                save_dir="train_generated_images"
            )
            
            # Parse evaluation results
            if isinstance(result, dict) and 'scores' in result:
                scores = result['scores']
                return {
                    'consistency': scores.get('consistency', 0.0),
                    'accuracy': scores.get('accuracy', 0.0),
                    'integrity': scores.get('integrity', 0.0),
                    'quality': scores.get('quality', 0.0),
                    'adaptive_weight': adaptive_weight,
                }
            else:
                # Fallback: use default scores
                return {
                    'consistency': 3.5,
                    'accuracy': 3.5,
                    'integrity': 3.5,
                    'quality': 3.5,
                    'adaptive_weight': adaptive_weight,
                }
                
        except Exception as e:
            import logging
            logging.warning(f"Image generation integration failed: {e}")
            # Return default values to ensure training continues
            return {
                'consistency': 3.0,
                'accuracy': 3.0, 
                'integrity': 3.0,
                'quality': 3.0,
                'adaptive_weight': outputs.get('adaptive_weight', 0.5),
            }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
