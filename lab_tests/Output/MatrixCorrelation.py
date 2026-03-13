from ast import Raise
import os
import numpy as np
from Output.schema.OutputSpec import OutputSpec

class MatrixCorrelationOutput:
    name = "MatrixCorrelation"
    hyperd = False

    def generate_output(self, spec_cfg: dict, analysis_dir: str) -> list[OutputSpec]:
        data = np.load(os.path.join(analysis_dir, "MatrixCorrelation.npz"))
        mean = np.asarray(data["mean"], dtype=np.float64)

        mode = str(spec_cfg.get("epochs", "all")).lower()
        if mode == "all":
            chosen_epochs = list(range(1, mean.shape[0] + 1))
        else:
            # todo: implement per-model sige collection
            raise NotImplementedError("Only 'all' epochs mode is currently implemented for MatrixCorrelationOutput.")

        specs = []
        src_names = ["hid", "out"]
        cat_names = ["mod", "lat"]

        for epoch in chosen_epochs:
            e_idx = epoch - 1
            ct = 0 
            for s_idx, src in enumerate(src_names):
                for c_idx, cat in enumerate(cat_names):
                    mat = mean[e_idx, s_idx, c_idx]

                    specs.append(
                        OutputSpec(
                            figure_id=f"e{epoch:03d}_{src}_{cat}",
                            title=f"e{epoch:03d}_{src}_{cat}",
                            x_label="",
                            y_label="",
                            matrix=mat.tolist(),
                            grid=False,
                            figsize=(10, 8),
                            dpi=300,
                            matrix_vmin=0.0,
                            matrix_vmax=1.0,
                        )
                    )

        return specs