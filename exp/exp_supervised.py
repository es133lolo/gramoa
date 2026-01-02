from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import wandb
import inspect
import os
import random
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from utils.tools import calculate_subject_level_metrics
from utils.losses import SupConRouterLoss, LoadBalancingLoss    

warnings.filterwarnings("ignore")

ProtoDict = Dict[int, Tuple[torch.Tensor, int]]


class Exp_Supervised(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa
        self.global_subject_prototypes: ProtoDict = {}
        self.router_contrastive = SupConRouterLoss(temperature=0.1) #temperature 체크 필요
        # ★ [추가] Load Balancing Loss 초기화 (전문가 수에 맞춰 설정)
        self.aux_loss_fn = LoadBalancingLoss(num_experts=args.n_experts)

    # warm-up schedule
    def schedule_lambda_con(self, epoch):
        warmup_epochs = self.args.warmup_con_epochs
        max_lambda = self.args.lambda_con
        if epoch < warmup_epochs:
            return 0.0
        return max_lambda * min(1.0, (epoch - warmup_epochs) / 5)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        # test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = train_data.max_seq_len  # redefine seq_len
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = train_data.X.shape[2]  # redefine enc_in
        self.args.num_class = len(np.unique(train_data.y[:, 0]))  # column 0 is the label
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        ids = []
        dataset_ids = []
        # which model to use
        model_to_use = self.swa_model if self.swa else self.model
        model_to_use.eval()

        with torch.no_grad():
            for i, (batch_x, label_id, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                label = label_id[:, 0].to(self.device)
                sub_id = label_id[:, 1].to(self.device)
                dataset_id = label_id[:, 2].to(self.device)

                B, T, C = batch_x.shape

                # dummy markers
                x_mark_enc = torch.zeros(B, T, 4, device=self.device)
                x_dec      = torch.zeros(B, T, C, device=self.device)
                x_mark_dec = torch.zeros(B, T, 4, device=self.device)

                # inspect signature
                sig = inspect.signature(model_to_use.forward).parameters
                supports_router = "return_router_repr" in sig

                # forward (unified for all models)
                # validation/test에서는 router loss 사용하지 않음
                if supports_router:
                    outputs = model_to_use(
                        batch_x,
                        x_mark_enc,
                        x_dec,
                        x_mark_dec,
                        return_router_repr=False
                    )
                else:
                    outputs = model_to_use(
                        batch_x,
                        x_mark_enc,
                        x_dec,
                        x_mark_dec
                    )

                if isinstance(outputs, tuple):
                     outputs = outputs[0]
                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)
                ids.append(sub_id)
                dataset_ids.append(dataset_id)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        ids = torch.cat(ids, 0)
        dataset_ids = torch.cat(dataset_ids, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        ids = ids.flatten().cpu().numpy()
        dataset_ids = dataset_ids.flatten().cpu().numpy()

        dataset_sample_results_list = []
        dataset_subject_results_list = []
        # calculate weighted metrics among datasets
        for dataset_id in np.unique(dataset_ids):
            mask = dataset_ids == dataset_id  # mask for samples from the same dataset

            sample_metrics_dict = {
                "Accuracy": accuracy_score(trues[mask], predictions[mask]),
                # "Precision": precision_score(trues[mask], predictions[mask], average="macro"),
                # "Recall": recall_score(trues[mask], predictions[mask], average="macro"),
                # "F1": f1_score(trues[mask], predictions[mask], average="macro"),
                # "AUROC": roc_auc_score(trues_onehot[mask], probs[mask], multi_class="ovr"),
                # "AUPRC": average_precision_score(trues_onehot[mask], probs[mask], average="macro"),
            }  # sample-level performance metrics
            # Check how many unique classes are present in the true labels
            unique_labels = np.unique(trues[mask])
            if len(unique_labels) < 2:
                # If there is only one class(e,g, leave-one-subject-out validation),
                # sample-level precision, recall, F1, AUROC and AUPRC are meaningless
                sample_metrics_dict["Precision"] = -1
                sample_metrics_dict["Recall"] = -1
                sample_metrics_dict["F1"] = -1
                sample_metrics_dict["AUROC"] = -1
                sample_metrics_dict["AUPRC"] = -1
            else:
                sample_metrics_dict["Precision"] = precision_score(trues[mask], predictions[mask], average="macro")
                sample_metrics_dict["Recall"] = recall_score(trues[mask], predictions[mask], average="macro")
                sample_metrics_dict["F1"] = f1_score(trues[mask], predictions[mask], average="macro")
                # AUROC
                try:
                    sample_metrics_dict["AUROC"] = roc_auc_score(
                        trues_onehot[mask], probs[mask], multi_class="ovr"
                    )
                except ValueError:
                    sample_metrics_dict["AUROC"] = float("nan")

                # AUPRC
                try:
                    sample_metrics_dict["AUPRC"] = average_precision_score(
                        trues_onehot[mask], probs[mask], average="macro"
                    )
                except ValueError:
                    sample_metrics_dict["AUPRC"] = float("nan")

            dataset_sample_results_list.append(sample_metrics_dict)

            subject_metrics_dict = calculate_subject_level_metrics(
                predictions[mask], trues[mask], ids[mask], self.args.num_class
            )  # subject-level performance metrics, do voting for each subject
            dataset_subject_results_list.append(subject_metrics_dict)

        def get_average_metrics(metrics_list):
            average_metrics = {key: 0 for key in metrics_list[0]}
            for metrics in metrics_list:
                for key, value in metrics.items():
                    average_metrics[key] += value
            for key in average_metrics:
                average_metrics[key] /= len(metrics_list)
            return average_metrics

        average_sample_metrics = get_average_metrics(dataset_sample_results_list)
        average_subject_metrics = get_average_metrics(dataset_subject_results_list)

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, average_sample_metrics, average_subject_metrics

    def train(self, setting):
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            name=setting,
            config=self.args.__dict__,
            reinit=True
        )   
            
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print("Training data shape: ", train_data.X.shape)
        print("Training label shape: ", train_data.y.shape)
        print("Validation data shape: ", vali_data.X.shape)
        print("Validation label shape: ", vali_data.y.shape)
        print("Test data shape: ", test_data.X.shape)
        print("Test label shape: ", test_data.y.shape)

        path = (
            "./checkpoints/"
            + self.args.method
            + "/"
            + self.args.task_name
            + "/"
            + self.args.model
            + "/"
            + self.args.model_id
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs)
        criterion = self._select_criterion()


        
        for epoch in range(self.args.train_epochs):
            # ======================================================
            # [분석 C/D] Router latent 저장용 buffer (epoch 단위)
            # ======================================================
            latent_buffer = {
                "latent": [],
                "expert": [],
                "spatial_expert": [],
                "diagnosis": [],
                "subject": [],
                "granularity": [],
            }

    
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()



            # ================================
            # [분석1] expert gate 통계 초기화
            # ================================
            num_classes = self.args.num_class

            gate_stats = {
                "temporal_sum": None,   # [C, E]
                "temporal_cnt": torch.zeros(num_classes, device="cpu"),
                "spatial_sum": None,    # [C, E]
                "spatial_cnt": torch.zeros(num_classes, device="cpu"),
            }


            for i, (batch_x, label_id, padding_mask) in enumerate(train_loader):
                t_gates = None
                s_gates = None
                
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label_id[:, 0]
                ids = label_id[:, 1]
                label = label.to(self.device)
                ids = ids.to(self.device)

                B, T, C = batch_x.shape
                # EEG에는 timestamp가 없으므로 zero dummy 사용
                x_mark_enc = torch.zeros(B, T, 4, device=self.device)
                x_dec      = torch.zeros(B, T, C, device=self.device)
                x_mark_dec = torch.zeros(B, T, 4, device=self.device)

                sig = inspect.signature(self.model.forward).parameters
                supports_router = "return_router_repr" in sig

                # -------------------------
                # Forward 패스
                # -------------------------
                if supports_router:
                    # ADformer/MoA 모델, router_probs는 게이팅 확률
                    
                    logits, router_repr, router_probs = self.model(
                        batch_x,
                        x_mark_enc,
                        x_dec,
                        x_mark_dec,
                        return_router_repr=True,
                        return_router_probs=True
                    )
                    # ======================================================
                    # [분석 C/D] Router latent 수집
                    # ======================================================
                    if router_repr is not None and router_probs is not None:
                        # router_repr: [B, G, D]
                        B, G, D = router_repr.shape
                        
                        # -------- Temporal gates (main expert axis) --------
                        t_gates = None
                        if ("temporal" in router_probs) and (router_probs["temporal"] is not None):
                            t_gates = router_probs["temporal"].get("gates", None)
                            if t_gates is not None and t_gates.dim() == 4:
                                t_gates = t_gates.mean(dim=0)  # [B,G,E]

                        # -------- Spatial gates (aux axis) --------
                        s_gates = None
                        if ("spatial" in router_probs) and (router_probs["spatial"] is not None):
                            s_gates = router_probs["spatial"].get("gates", None)
                            if s_gates is not None and s_gates.dim() == 4:
                                s_gates = s_gates.mean(dim=0)  # [B,G,E]

                        if t_gates is not None:
                            # ===== expert assignment =====
                            temporal_expert = t_gates.argmax(dim=-1)  # [B,G]

                            if s_gates is not None:
                                if s_gates.dim() == 3:
                                    spatial_expert = s_gates.argmax(dim=-1)  # [B,G]
                                elif s_gates.dim() == 2:
                                    spatial_expert = s_gates.argmax(dim=-1).unsqueeze(1).expand(B, G)
                            else:
                                spatial_expert = torch.full_like(temporal_expert, -1)

                            # ===== flatten =====
                            latent_flat = router_repr.detach().cpu().reshape(B * G, D)
                            temporal_flat = temporal_expert.reshape(-1).cpu()
                            spatial_flat  = spatial_expert.reshape(-1).cpu()
                            diag_flat = label.unsqueeze(1).expand(B, G).reshape(-1).cpu()
                            subj_flat = ids.unsqueeze(1).expand(B, G).reshape(-1).cpu()
                            gran_flat = torch.arange(G).repeat(B)

                            # ===== buffer append (single, clean) =====
                            latent_buffer["latent"].append(latent_flat)
                            latent_buffer["expert"].append(temporal_flat)          # temporal expert
                            latent_buffer["spatial_expert"].append(spatial_flat)   # spatial expert
                            latent_buffer["diagnosis"].append(diag_flat)
                            latent_buffer["subject"].append(subj_flat)
                            latent_buffer["granularity"].append(gran_flat)

                else:
                    # Conformer, Medformer, EEG-Conformer 계열
                    logits = self.model(
                        batch_x,
                        x_mark_enc,
                        x_dec,
                        x_mark_dec
                    )
                    router_repr = None
                    router_probs = None


                ce_loss = criterion(logits, label.long())

                # ★ contrastive label: subject_id + granularity
                if router_repr is not None:
                    B, G, D = router_repr.shape
                    gran_ids = torch.arange(G, device=self.device).unsqueeze(0).expand(B, G)
                    contrast_labels = ids.unsqueeze(1) * 50 + gran_ids

                    lambda_con = self.schedule_lambda_con(epoch)
                    con_loss = self.router_contrastive(router_repr, contrast_labels)
                else:
                    con_loss = torch.tensor(0.0, device=self.device)
                    lambda_con = 0.0

                # ★ [추가] Load Balancing Loss 계산
                
                # lambda_aux: 밸런싱 로스 가중치 (보통 0.01 ~ 0.1 사용)
                lambda_aux = getattr(self.args, "lambda_aux", 0.0)
                aux_loss = torch.tensor(0.0, device=self.device)

                # ----------------------------
                # 권장 안전 가드 (router_probs 구조/shape 방어)
                # ----------------------------
                if supports_router and (router_probs is not None):
                    if not isinstance(router_probs, dict):
                        # 예상 구조가 아니면 aux를 0으로 두고 진행 (실험이 죽지 않게)
                        router_probs = None

                if supports_router and (router_probs is not None):
                    # Temporal gates
                    t_gates = None
                    if ("temporal" in router_probs) and (router_probs["temporal"] is not None):
                        t_gates = router_probs["temporal"].get("gates", None)

                    # Spatial gates
                    s_gates = None
                    if ("spatial" in router_probs) and (router_probs["spatial"] is not None):
                        s_gates = router_probs["spatial"].get("gates", None)

                    # gates shape/dtype 가드 
                    # - 기대: [L,B,G,E] 또는 [B,G,E] 또는 [*,E]
                    if t_gates is not None:
                        if not torch.is_floating_point(t_gates):
                            t_gates = t_gates.float()
                        aux_loss = aux_loss + self.aux_loss_fn(t_gates)

                    if s_gates is not None:
                        if not torch.is_floating_point(s_gates):
                            s_gates = s_gates.float()
                        aux_loss = aux_loss + self.aux_loss_fn(s_gates)
                
                # ================================
                # [분석1] class별 expert 사용량 누적
                # ================================

                def normalize_gates(gates):
                    """
                    gates:
                    [L,B,G,E] -> [B,E]
                    [B,G,E]   -> [B,E]
                    [B,E]     -> [B,E]
                    """
                    if gates.dim() == 4:
                        gates = gates.mean(dim=0)    # L 평균
                        gates = gates.mean(dim=1)    # G 평균
                    elif gates.dim() == 3:
                        gates = gates.mean(dim=1)    # G 평균
                    return gates.detach()            # [B,E]

                # ---- Temporal MoA ----
                if t_gates is not None:
                    tg = normalize_gates(t_gates)   # [B,E]
                    B, E = tg.shape

                    if gate_stats["temporal_sum"] is None:
                        gate_stats["temporal_sum"] = torch.zeros(num_classes, E, device="cpu")

                    for c in range(num_classes):
                        mask = (label == c)
                        if mask.any():
                            gate_stats["temporal_sum"][c] += tg[mask].sum(dim=0).cpu()
                            gate_stats["temporal_cnt"][c] += mask.sum()

                # ---- Spatial MoA ----
                if s_gates is not None:
                    sg = normalize_gates(s_gates)    # [B,E] # “공간 granularity 평균 후 spatial expert 사용량. Spatial routing preference aggregated across channel groups
                    B, E = sg.shape

                    if gate_stats["spatial_sum"] is None:
                        gate_stats["spatial_sum"] = torch.zeros(num_classes, E, device="cpu")

                    for c in range(num_classes):
                        mask = (label == c)
                        if mask.any():
                            gate_stats["spatial_sum"][c] += sg[mask].sum(dim=0).cpu()
                            gate_stats["spatial_cnt"][c] += mask.sum()


                # 최종 Loss 합산
                loss = ce_loss + (lambda_con * con_loss) + (lambda_aux * aux_loss)
                

                train_loss.append(loss.item())

                wandb.log({
                    "total_loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "con_loss": con_loss.item(),
                    "aux_loss": aux_loss.item(),
                    "lambda_con": lambda_con
                })

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            # ======================================================
            # [분석 C/D] Router latent epoch 단위 저장
            # ======================================================
            if len(latent_buffer["latent"]) > 0:
                latent_save_dir = os.path.join(path, "router_latent_analysis")
                os.makedirs(latent_save_dir, exist_ok=True)

                torch.save({
                    "latent": torch.cat(latent_buffer["latent"], dim=0),
                    "expert": torch.cat(latent_buffer["expert"], dim=0),
                    "spatial_expert": torch.cat(latent_buffer["spatial_expert"], dim=0),
                    "diagnosis": torch.cat(latent_buffer["diagnosis"], dim=0),
                    "subject": torch.cat(latent_buffer["subject"], dim=0),
                    "granularity": torch.cat(latent_buffer["granularity"], dim=0),
                }, os.path.join(latent_save_dir, f"router_latent_epoch{epoch}.pt"))



            # ================================
            # [분석1] epoch 단위 expert 사용 비율 계산
            # ================================

            save_dir = os.path.join(path, "expert_analysis")
            os.makedirs(save_dir, exist_ok=True)

            def compute_avg(sum_tensor, cnt_tensor):
                out = sum_tensor.clone()
                for c in range(out.shape[0]):
                    if cnt_tensor[c] > 0:
                        out[c] /= cnt_tensor[c]
                return out
            
            if gate_stats["temporal_sum"] is not None:
                temporal_avg = compute_avg(
                    gate_stats["temporal_sum"],
                    gate_stats["temporal_cnt"]
                )  # [C,E]

                torch.save(
                    temporal_avg,
                    os.path.join(save_dir, f"temporal_expert_usage_epoch{epoch}.pt")
                )

                # ================================
                # [분석1] WandB Table: Class × Expert (Temporal)
                # ================================
                num_classes, num_experts = temporal_avg.shape

                class_names = [f"Class_{c}" for c in range(num_classes)]
                expert_names = [f"Expert_{e}" for e in range(num_experts)]

                temporal_table = wandb.Table(
                    columns=["Class"] + expert_names
                )

                for c in range(num_classes):
                    row = [class_names[c]] + temporal_avg[c].tolist()
                    temporal_table.add_data(*row)

                wandb.log({
                    "Temporal_Expert_Usage_Table": temporal_table,
                    "epoch": epoch
                })

                # wandb log (flatten해서 히스토그램)
                wandb.log({
                    "temporal_expert_usage": wandb.Histogram(
                        temporal_avg.flatten().numpy()
                    )
                })

            if gate_stats["spatial_sum"] is not None:
                spatial_avg = compute_avg(
                    gate_stats["spatial_sum"],
                    gate_stats["spatial_cnt"]
                )  # [C,E]

                torch.save(
                    spatial_avg,
                    os.path.join(save_dir, f"spatial_expert_usage_epoch{epoch}.pt")
                )

                # ================================
                # WandB Table: Class × Expert (Spatial)
                # ================================
                num_classes, num_experts = spatial_avg.shape

                class_names = [f"Class_{c}" for c in range(num_classes)]
                expert_names = [f"Expert_{e}" for e in range(num_experts)]

                spatial_table = wandb.Table(
                    columns=["Class"] + expert_names
                )

                for c in range(num_classes):
                    row = [class_names[c]] + spatial_avg[c].tolist()
                    spatial_table.add_data(*row)

                wandb.log({
                    "Spatial_Expert_Usage_Table": spatial_table,
                    "epoch": epoch
                })
                wandb.log({
                    "spatial_expert_usage": wandb.Histogram(
                        spatial_avg.flatten().numpy()
                    )
                })


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, sample_val_metrics_dict, subject_val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, sample_test_metrics_dict, subject_test_metrics_dict = self.vali(test_data, test_loader, criterion)

            # epoch 곡선 보기
            wandb.log({
                    # --------------------------
                    # Training / Validation Loss
                    # --------------------------
                    "train_loss_epoch": np.mean(train_loss),
                    "val_loss": vali_loss,
                    "test_loss": test_loss,

                    # --------------------------
                    # Sample-level metrics
                    # --------------------------
                    "val_accuracy": sample_val_metrics_dict["Accuracy"],
                    "val_f1": sample_val_metrics_dict["F1"],
                    "test_accuracy": sample_test_metrics_dict["Accuracy"],
                    "test_f1": sample_test_metrics_dict["F1"],

                    # --------------------------
                    # Subject-level metrics
                    # --------------------------
                    "val_subj_accuracy": subject_val_metrics_dict["Accuracy"],
                    "val_subj_f1": subject_val_metrics_dict["F1"],
                    "test_subj_accuracy": subject_test_metrics_dict["Accuracy"],
                    "test_subj_f1": subject_test_metrics_dict["F1"],
                })


            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch: {epoch + 1}, "
                f"Steps: {train_steps}, | Train Loss: {train_loss:.5f} | Learning Rate: {current_lr:.5e}\n"
                
                f"Sample-level results: \n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {sample_val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {sample_val_metrics_dict['Precision']:.5f}, "
                f"Recall: {sample_val_metrics_dict['Recall']:.5f}, "
                f"F1: {sample_val_metrics_dict['F1']:.5f}, "
                f"AUROC: {sample_val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {sample_val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {sample_test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {sample_test_metrics_dict['Precision']:.5f}, "
                f"Recall: {sample_test_metrics_dict['Recall']:.5f} "
                f"F1: {sample_test_metrics_dict['F1']:.5f}, "
                f"AUROC: {sample_test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {sample_test_metrics_dict['AUPRC']:.5f}\n"
                
                f"Subject-level results after majority voting: \n"
                f"Validation results, "
                f"Accuracy: {subject_val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {subject_val_metrics_dict['Precision']:.5f}, "
                f"Recall: {subject_val_metrics_dict['Recall']:.5f}, "
                f"F1: {subject_val_metrics_dict['F1']:.5f}, "
                f"AUROC: {subject_val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {subject_val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results, "
                f"Accuracy: {subject_test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {subject_test_metrics_dict['Precision']:.5f}, "
                f"Recall: {subject_test_metrics_dict['Recall']:.5f} "
                f"F1: {subject_test_metrics_dict['F1']:.5f}, "
                f"AUROC: {subject_test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {subject_test_metrics_dict['AUPRC']:.5f}\n"
            )
            early_stopping(
                -sample_val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.step()

        best_model_path = path + "checkpoint.pth"
        """# for swa model, to best leverage the functionality of stochastic weight average,
        # we take the last model as the final model,
        # which is the model after args.patience epochs (early stop counting) of best model on validation set"""
        if self.swa:
            """try:
                print("Saving the last model to leverage the functionality of stochastic weight average")
                torch.save(self.swa_model.state_dict(), best_model_path)
            except Exception as e:
                print(f"Error saving model: {e}")"""
            self.swa_model.load_state_dict(torch.load(best_model_path))
        # for normal model, we simply take the best model on validation set
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.method
                + "/"
                + self.args.task_name
                + "/"
                + self.args.model
                + "/"
                + self.args.model_id
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))
            print(f"Loading model successful at {model_path}, start to test")

        total_params = sum(p.numel() for p in self.model.parameters())
        criterion = self._select_criterion()

        def _run_forward(model, batch_x):
            B, T, C = batch_x.shape

            # Dummy timestamps (EEG에는 시간 정보를 쓰지 않음)
            x_mark_enc = torch.zeros(B, T, 4, device=self.device)
            x_dec = torch.zeros(B, T, C, device=self.device)
            x_mark_dec = torch.zeros(B, T, 4, device=self.device)

            sig = inspect.signature(model.forward).parameters
            supports_router = "return_router_repr" in sig

            if supports_router:
                outputs = model(
                    batch_x,
                    x_mark_enc,
                    x_dec,
                    x_mark_dec,
                    return_router_repr=False
                )
            else:
                outputs = model(
                    batch_x,
                    x_mark_enc,
                    x_dec,
                    x_mark_dec
                )

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return outputs
    
        vali_loss, sample_val_metrics_dict, subject_val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, sample_test_metrics_dict, subject_test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
            "./results/"
            + self.args.method
            + "/"
            + self.args.task_name
            + "/"
            + self.args.model
            + "/"
            + self.args.model_id
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Sample-level results: \n"
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {sample_val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {sample_val_metrics_dict['Precision']:.5f}, "
            f"Recall: {sample_val_metrics_dict['Recall']:.5f}, "
            f"F1: {sample_val_metrics_dict['F1']:.5f}, "
            f"AUROC: {sample_val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {sample_val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {sample_test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {sample_test_metrics_dict['Precision']:.5f}, "
            f"Recall: {sample_test_metrics_dict['Recall']:.5f} "
            f"F1: {sample_test_metrics_dict['F1']:.5f}, "
            f"AUROC: {sample_test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {sample_test_metrics_dict['AUPRC']:.5f}\n"

            f"Subject-level results after majority voting: \n"
            f"Validation results, "
            f"Accuracy: {subject_val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {subject_val_metrics_dict['Precision']:.5f}, "
            f"Recall: {subject_val_metrics_dict['Recall']:.5f}, "
            f"F1: {subject_val_metrics_dict['F1']:.5f}, "
            f"AUROC: {subject_val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {subject_val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results, "
            f"Accuracy: {subject_test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {subject_test_metrics_dict['Precision']:.5f}, "
            f"Recall: {subject_test_metrics_dict['Recall']:.5f} "
            f"F1: {subject_test_metrics_dict['F1']:.5f}, "
            f"AUROC: {subject_test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {subject_test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "results.txt"
        file_path = os.path.join(folder_path, file_name)
        f = open(file_path, "a")
        f.write("Model Setting: " + setting + "  \n")
        f.write(
            f"Sample-level results: \n"
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {sample_val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {sample_val_metrics_dict['Precision']:.5f}, "
            f"Recall: {sample_val_metrics_dict['Recall']:.5f}, "
            f"F1: {sample_val_metrics_dict['F1']:.5f}, "
            f"AUROC: {sample_val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {sample_val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {sample_test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {sample_test_metrics_dict['Precision']:.5f}, "
            f"Recall: {sample_test_metrics_dict['Recall']:.5f} "
            f"F1: {sample_test_metrics_dict['F1']:.5f}, "
            f"AUROC: {sample_test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {sample_test_metrics_dict['AUPRC']:.5f}\n"

            f"Subject-level results after majority voting: \n"
            f"Validation results, "
            f"Accuracy: {subject_val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {subject_val_metrics_dict['Precision']:.5f}, "
            f"Recall: {subject_val_metrics_dict['Recall']:.5f}, "
            f"F1: {subject_val_metrics_dict['F1']:.5f}, "
            f"AUROC: {subject_val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {subject_val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results, "
            f"Accuracy: {subject_test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {subject_test_metrics_dict['Precision']:.5f}, "
            f"Recall: {subject_test_metrics_dict['Recall']:.5f} "
            f"F1: {subject_test_metrics_dict['F1']:.5f}, "
            f"AUROC: {subject_test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {subject_test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.close()

        return (sample_val_metrics_dict, subject_val_metrics_dict,
                sample_test_metrics_dict, subject_test_metrics_dict,
                total_params)
