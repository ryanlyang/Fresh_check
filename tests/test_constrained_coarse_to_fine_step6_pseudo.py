from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from jetclass_fresh.hlt_cache import (
    DEFAULT_HLT_SEEDS,
    HLT_PROFILE_V2_REALISTIC,
    fixed_hlt_params_from_profile,
    generate_and_cache_hlt_view,
)
from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    JetIdentity,
    JetView,
    LABEL_NAMES,
    RAW_TOKEN_DIM,
    SPLIT_ORDER,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.constrained_coarse_to_fine import (
    ACCOUNTING_FIELD_NAMES,
    C5_B1,
    C6_MULTIVIEW,
    COARSE_TO_FINE_TRAIN_CONTRACT,
    PID_CATEGORY_NAMES,
    PID_PT_INDICES,
    PSEUDO_PARTICLE_CACHE_CONTRACT,
    PseudoParticleCacheConfig,
    audit_pseudo_particle_cache,
    build_c_tier_reconstructor,
    cache_pseudo_particle_views,
    default_hierarchy_target_layout,
    hlt_reference_axis,
    load_coarse_to_fine_reconstructor_checkpoint,
    load_pseudo_particle_cache,
    render_pseudo_particle_batch,
)
from teacher_logit_reco.constrained_coarse_to_fine.pseudo import _generate_rendered


def _particle(pt: float, eta: float, phi: float, pid: int) -> np.ndarray:
    token = np.zeros(RAW_TOKEN_DIM, dtype=np.float32)
    token[0] = float(pt)
    token[1] = float(eta)
    token[2] = float(phi)
    token[3] = float(pt) * float(np.cosh(eta))
    token[4] = (-1.0, 0.0, 1.0)[pid % 3]
    token[5 + int(pid)] = 1.0
    return token


def _toy_view(split: str, split_index: int, n_jets: int = 4) -> JetView:
    tokens = np.zeros((n_jets, 8, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 8), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % len(LABEL_NAMES)
    identities = []
    for jet in range(n_jets):
        identities.append(JetIdentity(f"toy_{split}.root", jet, int(labels[jet])))
        mask[jet, :6] = True
        for particle in range(6):
            tokens[jet, particle] = _particle(
                30.0 - 2.0 * particle + 0.2 * jet,
                -0.20 + 0.075 * particle + 0.01 * split_index,
                -0.16 + 0.06 * particle,
                particle % len(PID_CATEGORY_NAMES),
            )
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=identities,
        split=split,
        metadata={"view": "offline"},
    )


def _small_model(variant: str):
    return build_c_tier_reconstructor(
        variant,
        hierarchy_overrides={
            "d_model": 32,
            "num_heads": 4,
            "encoder_layers": 1,
            "pool_layers": 1,
            "decoder_layers_per_level": 1,
            "ffn_multiplier": 2.0,
            "pair_hidden_dim": 8,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        slot_overrides={
            "ffn_multiplier": 2.0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        layout=default_hierarchy_target_layout(radial_boundary=0.16),
    )


def _part_inputs(view: JetView):
    part = build_particle_transformer_inputs_from_tokens(view.tokens, view.mask)
    return (
        torch.from_numpy(part.pf_points),
        torch.from_numpy(part.pf_features),
        torch.from_numpy(part.pf_vectors),
        torch.from_numpy(part.pf_mask),
    )


def _write_sources(root: Path) -> tuple[Path, Path, str]:
    views = {split: _toy_view(split, index) for index, split in enumerate(SPLIT_ORDER)}
    manifest = SplitManifest(
        data_dir="toy",
        max_constits=8,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: len(view.jet_ids) for split, view in views.items()},
        split_seeds={split: 901 + index for index, split in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits={split: list(view.jet_ids) for split, view in views.items()},
        metadata={"step6_test": True},
    )
    manifest_path = root / "split_manifest.json.gz"
    save_split_manifest(manifest, manifest_path)
    manifest_sha = manifest_hash(manifest)
    hlt_dir = root / "hlt"
    params = fixed_hlt_params_from_profile(HLT_PROFILE_V2_REALISTIC, 2.5)
    for split in ("model_val", "final_test"):
        view = views[split]
        view.metadata["source_manifest_hash"] = manifest_sha
        generate_and_cache_hlt_view(
            view,
            hlt_dir,
            seed=DEFAULT_HLT_SEEDS[split],
            params=params,
            hlt_degradation_strength=2.5,
        )
    return manifest_path, hlt_dir, manifest_sha


def _checkpoint_payload(model, manifest_sha: str, *, role: str = "best_model_val"):
    layout = model.hierarchy.layout.to_dict()
    provenance = {
        split: {
            "split": split,
            "source_manifest_hash": manifest_sha,
            "layout": layout,
        }
        for split in ("model_train", "model_val")
    }
    return {
        "checkpoint_contract": COARSE_TO_FINE_TRAIN_CONTRACT,
        "checkpoint_role": role,
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "model": {
            "family": "C",
            "variant": model.slot_decoder.config.variant,
            "hierarchy_config": model.hierarchy.config.to_dict(),
            "slot_config": model.slot_decoder.config.to_dict(),
        },
        "metrics": {"selection.reconstruction_score": 1.0},
        "provenance": provenance,
    }


class ConstrainedCoarseToFineStep6PseudoTests(unittest.TestCase):
    def test_rendered_slots_preserve_additive_accounting_and_expose_trust_fields(self):
        view = _toy_view("model_val", 1, n_jets=2)
        model = _small_model(C5_B1).eval()
        reference_eta, reference_phi, _ = hlt_reference_axis(view.tokens, view.mask)
        with torch.no_grad():
            output = model(*_part_inputs(view))
            rendered = render_pseudo_particle_batch(
                output,
                reference_eta=torch.from_numpy(reference_eta),
                reference_phi=torch.from_numpy(reference_phi),
                model=model,
            )
        self.assertEqual(rendered.tokens.shape, (2, 1, 8 * 17, RAW_TOKEN_DIM))
        self.assertEqual(rendered.slot_log_sigma.shape[-1], 5)
        self.assertTrue(torch.isfinite(rendered.tokens).all())
        self.assertEqual(int(rendered.token_is_dust.sum()), 8)
        target = output.slots.terminal_accounting[:, None]
        for cell in range(8):
            selected = rendered.token_cell_indices == cell
            tokens = rendered.tokens[:, :, selected]
            counts = rendered.expected_count[:, :, selected]
            torch.testing.assert_close(
                tokens[..., 0].sum(dim=-1),
                target[:, :, cell, ACCOUNTING_FIELD_NAMES.index("total_pT")],
                atol=3.0e-5,
                rtol=3.0e-5,
            )
            torch.testing.assert_close(
                tokens[..., 3].sum(dim=-1),
                target[:, :, cell, ACCOUNTING_FIELD_NAMES.index("total_energy")],
                atol=3.0e-5,
                rtol=3.0e-5,
            )
            torch.testing.assert_close(
                counts.sum(dim=-1),
                target[:, :, cell, ACCOUNTING_FIELD_NAMES.index("expected_constituent_count")],
                atol=3.0e-5,
                rtol=3.0e-5,
            )
            for category, index in enumerate(PID_PT_INDICES):
                torch.testing.assert_close(
                    (tokens[..., 0] * tokens[..., 5 + category]).sum(dim=-1),
                    target[:, :, cell, index],
                    atol=3.0e-5,
                    rtol=3.0e-5,
                )

    def test_c6_multiview_generation_is_seeded_and_single_view_models_fail_closed(self):
        view = _toy_view("model_val", 1, n_jets=2)
        inputs = _part_inputs(view)
        reference_eta, reference_phi, _ = hlt_reference_axis(view.tokens, view.mask)
        kwargs = dict(
            points=inputs[0],
            features=inputs[1],
            vectors=inputs[2],
            mask=inputs[3],
            reference_eta=torch.from_numpy(reference_eta),
            reference_phi=torch.from_numpy(reference_phi),
            num_views=4,
            min_particle_pt=0.0,
            dust_reliability=0.1,
        )
        model = _small_model(C6_MULTIVIEW).eval()
        with torch.no_grad():
            first = _generate_rendered(model, generator=torch.Generator().manual_seed(44), **kwargs)
            second = _generate_rendered(model, generator=torch.Generator().manual_seed(44), **kwargs)
        self.assertEqual(first.num_views, 4)
        torch.testing.assert_close(first.tokens, second.tokens)
        self.assertGreater(float((first.tokens[:, 0] - first.tokens[:, 1]).abs().mean()), 1.0e-7)
        with self.assertRaisesRegex(ValueError, "C6"):
            _generate_rendered(
                _small_model(C5_B1).eval(),
                generator=torch.Generator().manual_seed(44),
                **kwargs,
            )

    def test_checkpoint_loader_requires_model_val_selected_c_tier_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = _small_model(C5_B1)
            selected = root / "best.pt"
            last = root / "last.pt"
            torch.save(_checkpoint_payload(model, "manifest", role="best_model_val"), selected)
            torch.save(_checkpoint_payload(model, "manifest", role="last"), last)
            loaded, payload = load_coarse_to_fine_reconstructor_checkpoint(selected)
            self.assertEqual(loaded.slot_decoder.config.variant, C5_B1)
            self.assertEqual(payload["checkpoint_role"], "best_model_val")
            with self.assertRaisesRegex(ValueError, "best_model_val"):
                load_coarse_to_fine_reconstructor_checkpoint(last)

    def test_hlt_only_sharded_cache_round_trip_and_final_test_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, hlt_dir, manifest_sha = _write_sources(root)
            checkpoint = root / "best_model_val.pt"
            torch.save(_checkpoint_payload(_small_model(C5_B1), manifest_sha), checkpoint)
            output_dir = root / "pseudo"
            config = PseudoParticleCacheConfig(
                output_cache_dir=str(output_dir),
                manifest_path=str(manifest_path),
                hlt_cache_dir=str(hlt_dir),
                reconstructor_checkpoint=str(checkpoint),
                splits=("model_val", "final_test"),
                batch_size=2,
                shard_size=2,
                device="cpu",
                amp=False,
                cache_dtype="float32",
                confirm_final_test=True,
            )
            report = cache_pseudo_particle_views(config)
            self.assertTrue(report["ok"])
            cache = load_pseudo_particle_cache(output_dir, C5_B1, "final_test")
            self.assertEqual(cache.arrays["tokens"].shape[0], 4)
            self.assertEqual(cache.metadata["cache_contract"], PSEUDO_PARTICLE_CACHE_CONTRACT)
            self.assertTrue(cache.metadata["inference_consumes_hlt_only"])
            self.assertFalse(cache.metadata["offline_cache_loaded"])
            self.assertFalse(cache.metadata["offline_targets_loaded"])
            self.assertTrue(cache.metadata["final_test_teacher_free"])
            audit = audit_pseudo_particle_cache(
                output_dir,
                variant=C5_B1,
                split="final_test",
                manifest_path=manifest_path,
                hlt_cache_dir=hlt_dir,
                checkpoint_path=checkpoint,
            )
            self.assertTrue(audit["ok"], audit["problems"])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                cache_pseudo_particle_views(
                    PseudoParticleCacheConfig(
                        **{
                            **config.__dict__,
                            "cache_dtype": "float16",
                        }
                    )
                )

    def test_final_test_rendering_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(ValueError, "confirm-final-test"):
            PseudoParticleCacheConfig(
                output_cache_dir="out",
                manifest_path="manifest.json.gz",
                hlt_cache_dir="hlt",
                reconstructor_checkpoint="best.pt",
                splits=("final_test",),
            )


if __name__ == "__main__":
    unittest.main()
