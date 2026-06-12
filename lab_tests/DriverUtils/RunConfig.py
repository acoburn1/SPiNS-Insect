from dataclasses import dataclass
from Eval.Correlation import LossEvaluator
from configs.utils import resolve_cfgs, get_stub
import numpy as np
from decimal import Decimal
import DataHelper.utils as DataUtils
from DataHelper.Probe import build_probe
import time
import os
import json
import sys
from pathlib import Path
from Model.StandardModel import StandardModel
import DriverUtils.Visual as Visual
from torch import kl_div, nn
from DataHelper import SpecialDataLoader as SDL
from DriverUtils.Zarr import build_zarr_from_results
from DriverUtils.RMutils import get_reference_matrices_m_l
from DriverUtils.Organize import group_graphs_by_name
from Output.schema.PlotOutput import plot_output
from Output.schema.dependencies import get_dependencies
from Statistics.StatHelper import stats_over_models
from DriverUtils.RunMetadata import git_branch, git_commit, utc_now_iso, write_stage_metadata
from DriverUtils.TabularExport import export_regression_tables

@dataclass
class RunConfig:
    def __init__(self, args):
        self.visual = args.visual

        self.d_cfg, self.m_cfg, self.o_cfg, self.p_cfg, self.dir_cfg = resolve_cfgs(args, idv=True)
        assert not (self.d_cfg['num_total_trials'] / self.d_cfg['num_mod_trials'] == 2 and self.d_cfg['special_dl'] or self.d_cfg['num_total_trials'] / self.d_cfg['num_mod_trials'] != 2 and not self.d_cfg['special_dl']), "special data loader must be used when num mod and num lat trials ineq"

        if (self.visual):
            Visual.print_cfgs(resolve_cfgs(args))

        self.alt = self.d_cfg['alt']
        self.training_name = self.d_cfg['training_name']
        self.special_dl = self.d_cfg['special_dl']
        self.num_mod_trials = self.d_cfg['num_mod_trials']
        self.num_total_trials = self.d_cfg['num_total_trials']
        self.generate_rms = self.d_cfg['generate_rms']

        self.num_features = self.m_cfg['num_features']
        self.hidden_layer_range = np.arange(
            self.m_cfg['hidden_layer_range']['start'], 
            self.m_cfg['hidden_layer_range']['end'] + self.m_cfg['hidden_layer_range']['step'], 
            self.m_cfg['hidden_layer_range']['step'])
        self.learning_rate_range = self._decimal_inclusive_range(
            self.m_cfg['learning_rate']['start'],
            self.m_cfg['learning_rate']['end'],
            self.m_cfg['learning_rate']['step'],
        )
        self.training_epochs = self.m_cfg['num_epochs']
        self.eval_epochs = self.training_epochs + 1
        self.num_models = self.m_cfg['num_models']
        self.adam = self.m_cfg.get('adam', True)
        self.relu = self.m_cfg.get('relu', True)
        
        self.X_probe, self.probe_metadata, self.probe_index = build_probe(self.p_cfg)

        self.training_data_filename = f"{self.dir_cfg['training_data']}/{self.training_name}.csv"
        self.training_inputs, self.training_outputs = DataUtils.csv_training_data_to_numpy(self.training_data_filename, num_features=self.num_features)
        assert len(self.training_inputs) == self.num_total_trials, f"expected {self.num_total_trials} training trials but got {len(self.training_inputs)}. check data config"
        self.dataloader = DataUtils.get_dataloader(self.training_inputs, self.training_outputs) if not self.special_dl else SDL.SpecialDataLoader(self.training_inputs, self.training_outputs, self.num_mod_trials)

        self.modular_p_m_filename = f"{self.dir_cfg['reference_matrices']}/cooc-jaccard-mod.csv"
        self.lattice_p_m_filename = f"{self.dir_cfg['reference_matrices']}/cooc-jaccard-lat.csv" if not self.alt else "Data/ReferenceMatrices/cooc-jaccard-lat-alt.csv"
        self.mod_rm, self.lat_rm = get_reference_matrices_m_l(self.modular_p_m_filename, self.lattice_p_m_filename, self.num_mod_trials, self.training_inputs, self.generate_rms)
        self.result_dir = f"{self.dir_cfg['results']}/{get_stub(args)}"
        self.activations_dir = f"{self.result_dir}/ActivationData/{self.training_name}"
        self.analysis_dir = f"{self.result_dir}/AnalysisData/{self.training_name}"
        self.output_dir = f"{self.result_dir}/Output/{self.training_name}"

        self.dependencies = get_dependencies(self.o_cfg)


    def train(self):
        """
        Trains models for each combination of hidden layer size and learning rate,
        evaluates them on the probe, and saves the activations to zarr files.
        """
        pair_n = int(len(self.hidden_layer_range) * len(self.learning_rate_range))
        total_models = int(pair_n * self.num_models)

        vis = None
        if self.visual:
            vis = Visual.ModelVisualInfo(
                hls=int(self.hidden_layer_range[0]),
                lr=float(self.learning_rate_range[0]),
                model_n=total_models,
                epoch_n=int(self.training_epochs),
            )
            vis.start_pair()

        global_model_i = 0
        pair_i = 0

        stage_started = utc_now_iso()
        stage_success = False

        try:
            for HLS in self.hidden_layer_range:
                for LR in self.learning_rate_range:
                    activations_dir = self._add_suffix(self.activations_dir, HLS, LR)
                    results = []

                    if vis is not None:
                        vis.note("") # reset note
                        vis.hls = int(HLS)
                        vis.lr = float(LR)

                    for i in range(self.num_models):
                        if vis is not None:
                            vis.model_i = global_model_i

                        model = StandardModel(
                            num_features=self.num_features,
                            hidden_layer_size=HLS,
                            batch_size=self.num_total_trials,
                            num_epochs=self.training_epochs,
                            learning_rate=LR,
                            loss_fn=nn.BCEWithLogitsLoss(),
                            adam=self.adam,
                            relu=self.relu,
                        )

                        result = model.train_eval(
                            self.dataloader,
                            self.X_probe,
                            vis=vis,  
                        )
                        results.append(result)

                        global_model_i += 1
                    
                    if vis is not None:
                        vis.note("saving zarr")

                    build_zarr_from_results(f"{activations_dir}/activations.zarr", results)

                    pair_i += 1

            stage_success = True

        finally:
            if vis is not None:
                vis.progress_done()
                time.sleep(0.1)

            write_stage_metadata(
                result_dir=self.result_dir,
                stage_name="activation_data",
                stage_dir=self.activations_dir,
                started_at_utc=stage_started,
                finished_at_utc=utc_now_iso(),
                status="success" if stage_success else "failed",
                config_path=f"{self.result_dir}/config.json",
                details={
                    "training_name": self.training_name,
                    "num_models": int(self.num_models),
                    "hidden_layer_range": [int(v) for v in self.hidden_layer_range.tolist()],
                    "learning_rate_range":[float(v) for v in self.learning_rate_range.tolist()],
                },
            )

    def evaluate(self):
        """
        Evaluates activations and saves the results to npz files.
        sig_epoch data is organized as a binary mask of shape (M, E) indicating whether each epoch is significant or not.
        The rest of the data is saved as a dictionary with keys 'raw', 'mean', 'std', 'se', 'ci_lo', 'ci_hi', and 'n', where
        'raw' is always organized as one value/object per condition(s), per epoch, per model with shape (M, E, C1, ..., Cn, D) and 
        the rest of the keys are arrays of shape (E, C1, ..., Cn, D) with statistics computed across models
        """
        evaluators = self.dependencies.evaluation_fns
        sige = self.dependencies.sige
        wb_sige = self.dependencies.wb_sige

        pair_n = int(len(self.hidden_layer_range) * len(self.learning_rate_range))
        eval_n = int(len(evaluators) + (1 if sige else 0) + (1 if wb_sige else 0))

        vis = None
        if self.visual:
            vis = Visual.EvalVisualInfo(
                hls=int(self.hidden_layer_range[0]),
                lr=float(self.learning_rate_range[0]),
                pair_i=0,
                pair_n=pair_n,
                eval_n=eval_n,
                model_n=int(self.num_models),
                epoch_n=int(self.eval_epochs),
            )
            vis.start()

        pair_i = 0
        stage_started = utc_now_iso()
        stage_success = False

        try:
            for HLS in self.hidden_layer_range:
                for LR in self.learning_rate_range:
                    activations_dir = self._add_suffix(self.activations_dir, HLS, LR)
                    zarr_path = f"{activations_dir}/activations.zarr"

                    analysis_dir = self._add_suffix(self.analysis_dir, HLS, LR)
                    os.makedirs(analysis_dir, exist_ok=True)

                    if vis is not None:
                        vis.note("") # reset note
                        vis.hls = int(HLS)
                        vis.lr = float(LR)
                        vis.pair_i = int(pair_i)

                    for ev_i, evaluator in enumerate(evaluators):
                        if vis is not None:
                            vis.set_eval(evaluator.name, ev_i)

                        raw, metadata = evaluator.run(self, zarr_path, vis=vis)

                        stats = stats_over_models(raw)

                        save_kwargs = {
                            "raw": raw,
                            "mean": stats["mean"],
                            "std": stats["std"],
                            "se": stats["se"],
                            "ci_lo": stats["ci_lo"],
                            "ci_hi": stats["ci_hi"],
                            "n": stats["n"],
                        }

                        if metadata is not None:
                            save_kwargs["metadata"] = np.array(metadata, dtype=object)

                        np.savez(
                            f"{analysis_dir}/{evaluator.name}.npz",
                            **save_kwargs,
                        )

                        if sige and evaluator.name == "Correlation":
                            if vis is not None:
                                vis.set_eval("SigEpoch", ev_i + 1)
                                vis.note("vector pass")

                            sige_results, _ = sige.run(raw)
                            np.savez(f"{analysis_dir}/sige.npz", results=sige_results)

                            if vis is not None:
                                vis.fast_done()

                        if wb_sige and evaluator.name == "WithinVsBetweenCorrelation":
                            if vis is not None:
                                vis.set_eval("WbSigEpoch", ev_i + 1)
                                vis.note("vector pass")

                            wb_sige_results, _ = wb_sige.run(raw)
                            np.savez(f"{analysis_dir}/wb-sige.npz", results=wb_sige_results)

                            if vis is not None:
                                vis.fast_done()

                    pair_i += 1

            stage_success = True

        finally:
            if vis is not None:
                vis.close()
                time.sleep(0.1)

            write_stage_metadata(
                result_dir=self.result_dir,
                stage_name="analysis_data",
                stage_dir=self.analysis_dir,
                started_at_utc=stage_started,
                finished_at_utc=utc_now_iso(),
                status="success" if stage_success else "failed",
                config_path=f"{self.result_dir}/config.json",
                details={
                    "training_name": self.training_name,
                    "evaluators": [ev.name for ev in evaluators],
                    "sig_epoch_enabled": bool(sige),
                    "wb_sig_epoch_enabled": bool(wb_sige),
                },
            )

    def generate_output(self):
        hyperd_out_fns = self.dependencies.hyperd_output_fns
        out_fns = self.dependencies.output_fns
        cfgs = self.dependencies.cfgs

        stage_started = utc_now_iso()
        stage_success = False
        failure_count = 0

        def report_failure(output_name: str, context: str, exc: Exception):
            nonlocal failure_count
            failure_count += 1
            print(
                f"Output failed: {output_name} ({context}): {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

        try:
            grouped_output_names: set[str] = set()

            for fn in hyperd_out_fns:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.visual:
                    Visual.status(f"Saving {fn.name} output to {self.output_dir} ...")
                output_name = self._output_name_for_cfg(fn.name, cfgs[fn])
                try:
                    specs = fn.generate_output(cfgs[fn], self.analysis_dir)
                except Exception as exc:
                    report_failure(fn.name, "hyperparameter output", exc)
                    continue

                for spec in specs:
                    plot_output(spec, f"{self.output_dir}/{output_name}")

            for HLS in self.hidden_layer_range:
                for LR in self.learning_rate_range:
                    output_dir = self._add_suffix(self.output_dir, HLS, LR)
                    analysis_dir = self._add_suffix(self.analysis_dir, HLS, LR)
                    for fn in out_fns:
                        os.makedirs(output_dir, exist_ok=True)
                        if self.visual:
                            Visual.status(f"Saving {fn.name} output to {output_dir} ...")
                        output_name = self._output_name_for_cfg(fn.name, cfgs[fn])
                        try:
                            specs = fn.generate_output(cfgs[fn], analysis_dir)
                        except Exception as exc:
                            report_failure(fn.name, f"HLS={HLS}, LR={LR}", exc)
                            continue

                        if cfgs[fn].get("group", False):
                            grouped_output_names.add(output_name)

                        for spec in specs:
                            plot_output(spec, f"{output_dir}/{output_name}")

            if self.visual:
                Visual.status(f"Organizing files ...")
            group_graphs_by_name(Path(self.output_dir).parent, grouped_output_names)

            stage_success = failure_count == 0
            if failure_count:
                print(
                    f"Output generation completed with {failure_count} failure(s).",
                    file=sys.stderr,
                )
        finally:
            write_stage_metadata(
                result_dir=self.result_dir,
                stage_name="output",
                stage_dir=f"{self.result_dir}/Output",
                started_at_utc=stage_started,
                finished_at_utc=utc_now_iso(),
                status="success" if stage_success else "failed",
                config_path=f"{self.result_dir}/config.json",
                details={
                    "training_name": self.training_name,
                    "hyperd_output_functions": [fn.name for fn in hyperd_out_fns],
                    "output_functions": [fn.name for fn in out_fns],
                },
            )

    def generate_tabular_output(self):
        for HLS in self.hidden_layer_range:
            for LR in self.learning_rate_range:
                output_dir = self._add_suffix(self.output_dir, HLS, LR)
                analysis_dir = self._add_suffix(self.analysis_dir, HLS, LR)
                os.makedirs(output_dir, exist_ok=True)
                export_regression_tables(analysis_dir, output_dir)

    def save_configuration(self):
        os.makedirs(self.result_dir, exist_ok=True)
        cfg = {
            "data_config": self.d_cfg,
            "model_config": self.m_cfg,
            "output_config": self.o_cfg,
            "probe_config": self.p_cfg,
            "directory_config": self.dir_cfg,
            "generated_at_utc": utc_now_iso(),
            "git_branch": git_branch(),
            "git_commit": git_commit(),
        }

        with open(f"{self.result_dir}/config.json", "w") as f:
            json.dump(cfg, f, indent=2)


    @staticmethod
    def _decimal_inclusive_range(start: float, end: float, step: float) -> np.ndarray:
        start_d = Decimal(str(start))
        end_d = Decimal(str(end))
        step_d = Decimal(str(step))

        values = []
        current = start_d
        while current <= end_d:
            values.append(float(current))
            current += step_d

        return np.array(values, dtype=np.float64)

    @staticmethod
    def _add_suffix(d: str, hls: int, lr: float) -> str:
        lr_str = f"{lr}".replace(".", "p")
        return d + f"_hls{hls}_lr{lr_str}"

    @staticmethod
    def _output_name_for_cfg(fn_name: str, cfg: dict) -> str:
        tokens = [fn_name]

        corr_type = str(cfg.get("corr_type", "")).lower()
        if corr_type in ("standard", "wb"):
            tokens.append(corr_type)

        sige_type = str(cfg.get("sige_type", "")).lower()
        if sige_type in ("sig", "wb-sig"):
            suffix = "sige" if sige_type == "sig" else "wb-sige"
            tokens.append(suffix)
            return "_".join(tokens)

        epoch_mode = str(cfg.get("epochs", "")).lower()
        if epoch_mode in ("sig", "wb-sig"):
            suffix = "sige" if epoch_mode == "sig" else "wb-sige"
            tokens.append(suffix)
            return "_".join(tokens)
        if epoch_mode == "range":
            tokens.append("range")
            return "_".join(tokens)

        return "_".join(tokens)

def confirm_configuration():
    response = input("does the above configuration look correct? (y/n): ").strip().lower()
    if response != 'y':
        print("execution cancelled.")
        exit(0)
